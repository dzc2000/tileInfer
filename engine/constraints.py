"""Constrained / guided decoding.

Supports:

* ``logit_bias``        - additive bias per token id
* ``allowed_token_ids`` - hard allow-list (everything else masked to -inf)
* ``bad_token_ids``     - hard deny-list
* ``guided_choice``     - restrict output to one of a list of strings (trie)
* ``guided_json``       - force the output to be valid JSON (character FSM)
* ``guided_regex``      - force the output to match a regular expression (NFA)

Constraints are applied per-sequence on the *sharded* logits produced by the
model runner, so every class below receives ``vocab_offset`` / ``vocab_shard``
(the current TP rank's slice of the vocabulary) and must translate global token
ids into local indices itself.

JSON and regex constraints build a trie over the tokenizer vocabulary lazily
(the trie is cached per tokenizer), then walk that trie with a per-character
grammar state to enumerate the allowed token ids. Full-vocabulary masks are
memoized per grammar state so the cost is paid once per state, not once per
decode step.
"""
from __future__ import annotations

import re as _re

import torch


# --------------------------------------------------------------------------- #
# Tokenizer vocabulary trie
# --------------------------------------------------------------------------- #
class _TrieNode:
    __slots__ = ('children', 'ids')

    def __init__(self):
        self.children: dict[str, _TrieNode] = {}
        self.ids: list[int] = []


class _VocabIndex:
    """Trie over the tokenizer vocabulary (decoded token string -> token ids)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.root = _TrieNode()
        self._build()

    def _build(self):
        try:
            vocab_size = len(self.tokenizer)
        except Exception:
            vocab_size = getattr(self.tokenizer, 'vocab_size', 0)
        if vocab_size <= 0:
            return
        # batch_decode is dramatically faster than vocab_size individual decodes.
        ids = [[i] for i in range(vocab_size)]
        try:
            strings = self.tokenizer.batch_decode(
                ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            strings = [self.tokenizer.decode(i, skip_special_tokens=False)
                       for i in range(vocab_size)]
        special = set(getattr(self.tokenizer, 'all_special_ids', []) or [])
        for tid, s in enumerate(strings):
            if not s or tid in special:
                continue
            node = self.root
            for ch in s:
                node = node.children.setdefault(ch, _TrieNode())
            node.ids.append(tid)

    def collect_allowed(self, grammar) -> list[int]:
        """Enumerate token ids that are valid continuations of ``grammar``."""
        allowed: list[int] = []

        def dfs(node: _TrieNode, state):
            allowed.extend(node.ids)
            for ch, child in node.children.items():
                nxt = state.clone()
                if nxt.step(ch):
                    dfs(child, nxt)

        dfs(self.root, grammar)
        return allowed


_VOCAB_INDEX_CACHE: dict[int, _VocabIndex] = {}


def _get_vocab_index(tokenizer) -> _VocabIndex:
    key = id(tokenizer)
    idx = _VOCAB_INDEX_CACHE.get(key)
    if idx is None:
        idx = _VocabIndex(tokenizer)
        _VOCAB_INDEX_CACHE[key] = idx
    return idx


# --------------------------------------------------------------------------- #
# Regex engine (Thompson NFA)
# --------------------------------------------------------------------------- #
class _NFAState:
    __slots__ = ('eps', 'trans', 'accept')

    def __init__(self):
        self.eps: set[_NFAState] = set()
        self.trans: list[tuple] = []   # (predicate, target)
        self.accept: bool = False


_DIGITS = frozenset('0123456789')
_WORD = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
_SPACE = frozenset(' \t\n\r\f\v')

_ESCAPE_SETS = {
    'd': _DIGITS,
    'w': _WORD,
    's': _SPACE,
    'D': _DIGITS,
    'W': _WORD,
    'S': _SPACE,
    'n': '\n',
    't': '\t',
    'r': '\r',
    'f': '\f',
    'v': '\v',
}


def _epsilon_closure(states: set) -> frozenset:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for nxt in s.eps:
            if nxt not in closure:
                closure.add(nxt)
                stack.append(nxt)
    return frozenset(closure)


def _parse_class(pattern: str, i: int):
    """Parse a ``[...]`` character class starting at ``pattern[i] == '['``.

    Returns ``(predicate, new_i)`` where ``new_i`` is the index after ``]``.
    """
    i += 1
    negate = False
    if i < len(pattern) and pattern[i] == '^':
        negate = True
        i += 1
    items: set[str] = set()
    while i < len(pattern) and pattern[i] != ']':
        if pattern[i] == '\\' and i + 1 < len(pattern):
            c = pattern[i + 1]
            if c in _ESCAPE_SETS:
                val = _ESCAPE_SETS[c]
                if isinstance(val, str):
                    items.add(val)
                else:
                    items |= set(val)
                i += 2
                continue
            items.add(c)
            i += 2
            continue
        if i + 2 < len(pattern) and pattern[i + 1] == '-' and pattern[i + 2] != ']':
            start, end = pattern[i], pattern[i + 2]
            for code in range(ord(start), ord(end) + 1):
                items.add(chr(code))
            i += 3
            continue
        items.add(pattern[i])
        i += 1
    if i < len(pattern) and pattern[i] == ']':
        i += 1
    if negate:
        return (lambda ch, s=items: ch not in s), i
    return (lambda ch, s=items: ch in s), i


def _parse_escape(pattern: str, i: int):
    """Parse an escape at ``pattern[i] == '\\'``. Returns ``(predicate, new_i)``."""
    c = pattern[i + 1]
    if c in _ESCAPE_SETS:
        val = _ESCAPE_SETS[c]
        if isinstance(val, str):
            return (lambda ch, v=val: ch == v), i + 2
        return (lambda ch, v=val: ch in v), i + 2
    return (lambda ch, v=c: ch == v), i + 2


def _parse_atom(pattern: str, i: int):
    """Parse one atom. Returns ``(start_state, accept_state, new_i)``."""
    if i >= len(pattern):
        raise ValueError("unexpected end of regex")
    c = pattern[i]
    if c == '(':
        start, accept, i = _parse_alt(pattern, i + 1)
        if i >= len(pattern) or pattern[i] != ')':
            raise ValueError("unbalanced parenthesis in regex")
        return start, accept, i + 1
    if c == '[':
        pred, i = _parse_class(pattern, i)
    elif c == '\\':
        pred, i = _parse_escape(pattern, i)
    elif c == '.':
        pred = lambda ch: ch != '\n'  # noqa: E731
        i += 1
    elif c in '*+?{':
        raise ValueError(f"dangling quantifier '{c}' in regex")
    else:
        pred = (lambda ch, v=c: ch == v)
        i += 1

    start = _NFAState()
    accept = _NFAState()
    start.trans.append((pred, accept))
    return start, accept, i


def _parse_concat(pattern: str, i: int):
    """Parse a concatenation (no alternation). Returns ``(start, accept, i)``."""
    start = _NFAState()
    tail = start
    while i < len(pattern) and pattern[i] not in '|)':
        if pattern[i] in '*+?{':
            # quantifier applied to the previous atom: rebuild tail
            raise ValueError(f"unexpected quantifier '{pattern[i]}'")
        a_start, a_accept, i = _parse_atom(pattern, i)
        # apply postfix quantifiers
        while i < len(pattern) and pattern[i] in '*+?{':
            op = pattern[i]
            if op == '{':
                j = pattern.find('}', i)
                if j == -1:
                    raise ValueError("unbalanced '{' in regex")
                spec = pattern[i + 1:j]
                i = j + 1
                a_start, a_accept = _quantify(a_start, a_accept, spec)
            else:
                i += 1
                a_start, a_accept = _quantify(a_start, a_accept, op)
        tail.eps.add(a_start)
        tail = a_accept
    return start, tail, i


def _quantify(start, accept, op: str):
    """Apply a quantifier to a Thompson fragment. Returns new (start, accept)."""
    n_start = _NFAState()
    n_accept = _NFAState()

    if op == '*':
        n_start.eps.add(start)
        accept.eps.add(start)
        accept.eps.add(n_accept)
        n_start.eps.add(n_accept)
        return n_start, n_accept
    if op == '+':
        n_start.eps.add(start)
        accept.eps.add(start)
        accept.eps.add(n_accept)
        return n_start, n_accept
    if op == '?':
        n_start.eps.add(start)
        accept.eps.add(n_accept)
        n_start.eps.add(n_accept)
        return n_start, n_accept

    # {m}, {m,}, {m,n}
    m, n = _parse_repeat(op)
    if m < 0:
        raise ValueError(f"invalid quantifier {{{op}}}")

    # Chain ``m`` mandatory copies in series.
    cur = n_start
    for k in range(m):
        s, a = (start, accept) if k == 0 else _clone_fragment(start, accept)
        cur.eps.add(s)
        cur = a

    if n == -1:
        # {m,}: loop back for one-or-more of the remaining repetitions.
        cur.eps.add(start)
        cur.eps.add(n_accept)
    else:
        # {m,n}: stop after m, or add up to (n - m) optional copies.
        cur.eps.add(n_accept)
        for _ in range(n - m):
            s, a = _clone_fragment(start, accept)
            cur.eps.add(s)
            a.eps.add(n_accept)
            cur = a

    return n_start, n_accept


def _parse_repeat(spec: str):
    if ',' in spec:
        m_s, n_s = spec.split(',', 1)
        m = int(m_s) if m_s else 0
        n = int(n_s) if n_s else -1
        return m, n
    return int(spec), int(spec)


def _clone_fragment(start, accept):
    mapping = {}

    def clone(s):
        if s in mapping:
            return mapping[s]
        ns = _NFAState()
        mapping[s] = ns
        ns.accept = s.accept
        for pred, t in s.trans:
            ns.trans.append((pred, clone(t)))
        for e in s.eps:
            ns.eps.add(clone(e))
        return ns

    return clone(start), clone(accept)


def _parse_alt(pattern: str, i: int):
    """Parse an alternation. Returns ``(start, accept, i)``."""
    start = _NFAState()
    accept = _NFAState()
    while True:
        c_start, c_accept, i = _parse_concat(pattern, i)
        start.eps.add(c_start)
        c_accept.eps.add(accept)
        if i < len(pattern) and pattern[i] == '|':
            i += 1
            continue
        break
    return start, accept, i


def compile_regex(pattern: str):
    """Compile a regex into a Thompson NFA. Returns ``(start_state)``."""
    # Strip anchors; the guided matcher always requires a full match.
    if pattern.startswith('^'):
        pattern = pattern[1:]
    if pattern.endswith('$'):
        pattern = pattern[:-1]
    start, accept, i = _parse_alt(pattern, 0)
    if i != len(pattern):
        raise ValueError(f"failed to parse regex near index {i}: {pattern!r}")
    accept.accept = True
    return start


class RegexState:
    """Current NFA state set for a regex-guided sequence."""

    __slots__ = ('states', 'nfa_start')

    def __init__(self, nfa_start):
        self.nfa_start = nfa_start
        self.states = _epsilon_closure({nfa_start})

    def clone(self):
        s = RegexState(self.nfa_start)
        s.states = frozenset(self.states)
        return s

    def step(self, ch: str):
        nxt: set[_NFAState] = set()
        for st in self.states:
            for pred, target in st.trans:
                if pred(ch):
                    nxt.add(target)
        if not nxt:
            return None
        closure = _epsilon_closure(nxt)
        if not closure:
            return None
        self.states = closure
        return True

    def can_accept(self) -> bool:
        return any(s.accept for s in self.states)

    def state_key(self):
        return frozenset(self.states)


# --------------------------------------------------------------------------- #
# JSON grammar FSM
# --------------------------------------------------------------------------- #
class _NumberState:
    __slots__ = ('phase', 'saw_digit', 'allow_sign', 'done',
                 'frac_digit', 'exp_digit')

    def __init__(self):
        self.phase = 'int'
        self.saw_digit = False
        self.allow_sign = True
        self.done = False
        self.frac_digit = False
        self.exp_digit = False

    def clone(self):
        s = _NumberState()
        s.phase = self.phase
        s.saw_digit = self.saw_digit
        s.allow_sign = self.allow_sign
        s.done = self.done
        s.frac_digit = self.frac_digit
        s.exp_digit = self.exp_digit
        return s

    def step(self, ch: str) -> bool:
        if self.done:
            return False
        if self.allow_sign and ch == '-':
            self.allow_sign = False
            return True
        if self.phase == 'int':
            if ch.isdigit():
                self.saw_digit = True
                self.allow_sign = False
                return True
            if ch == '.' and self.saw_digit:
                self.phase = 'frac'
                return True
            if ch in 'eE' and self.saw_digit:
                self.phase = 'exp'
                return True
            if self.saw_digit:
                self.done = True
            return False
        if self.phase == 'frac':
            if ch.isdigit():
                self.frac_digit = True
                return True
            if ch in 'eE' and self.frac_digit:
                self.phase = 'exp'
                return True
            if self.frac_digit:
                self.done = True
            return False
        if self.phase == 'exp':
            if ch in '+-':
                self.phase = 'exp_sign'
                return True
            if ch.isdigit():
                self.phase = 'exp_digits'
                self.exp_digit = True
                return True
            return False
        if self.phase == 'exp_sign':
            if ch.isdigit():
                self.phase = 'exp_digits'
                self.exp_digit = True
                return True
            return False
        if self.phase == 'exp_digits':
            if ch.isdigit():
                self.exp_digit = True
                return True
            self.done = True
            return False
        return False

    def is_complete(self) -> bool:
        if self.done:
            return True
        if self.phase == 'int':
            return self.saw_digit
        if self.phase == 'frac':
            return self.frac_digit
        if self.phase == 'exp_digits':
            return self.exp_digit
        return False


class _LiteralState:
    __slots__ = ('target', 'pos', 'done')

    def __init__(self, first: str):
        self.target = {'t': 'true', 'f': 'false', 'n': 'null'}[first]
        self.pos = 1
        self.done = self.pos >= len(self.target)

    def clone(self):
        s = _LiteralState(self.target[0])
        s.target = self.target
        s.pos = self.pos
        s.done = self.done
        return s

    def step(self, ch: str) -> bool:
        if self.pos >= len(self.target):
            self.done = True
            return False
        if ch == self.target[self.pos]:
            self.pos += 1
            if self.pos >= len(self.target):
                self.done = True
            return True
        return False


class JsonGrammar:
    """Character-level FSM that validates a JSON document as it is generated."""

    def __init__(self):
        # Each stack entry is [kind, has_elems]: kind in {'obj','arr'}, and
        # has_elems records whether the container has any key/value yet (used
        # to reject trailing commas and allow empty containers).
        self.stack: list[list] = []
        self.expect = 'value'            # value|key|colon|key_end|arr_end|done
        self.in_str = False
        self.esc = False
        self.str_is_key = False
        self.num: _NumberState | None = None
        self.lit: _LiteralState | None = None
        self.complete = False

    def clone(self):
        g = JsonGrammar()
        g.stack = [list(e) for e in self.stack]
        g.expect = self.expect
        g.in_str = self.in_str
        g.esc = self.esc
        g.str_is_key = self.str_is_key
        g.num = self.num.clone() if self.num is not None else None
        g.lit = self.lit.clone() if self.lit is not None else None
        g.complete = self.complete
        return g

    def step(self, ch: str) -> bool:
        if self.complete:
            return ch in ' \t\n\r'
        if self.in_str:
            if self.esc:
                self.esc = False
                return True
            if ch == '\\':
                self.esc = True
                return True
            if ch == '"':
                self.in_str = False
                self._after_string()
                return True
            if ord(ch) < 0x20:
                return False
            return True

        if ch in ' \t\n\r':
            return True

        if self.num is not None:
            if self.num.step(ch):
                if self.num.done:
                    self.num = None
                    self._after_value()
                return True
            self.num = None
            self._after_value()
        if self.lit is not None:
            if self.lit.step(ch):
                if self.lit.done:
                    self.lit = None
                    self._after_value()
                return True
            self.lit = None
            self._after_value()

        return self._structural(ch)

    def _structural(self, ch: str) -> bool:
        e = self.expect
        if e == 'value':
            return self._start_value(ch)
        if e == 'key':
            if ch == '}':
                # only valid for an empty object (no key started yet)
                if self.stack and not self.stack[-1][1]:
                    self.stack.pop()
                    self._after_value()
                    return True
                return False
            if ch == '"':
                self.stack[-1][1] = True
                self.in_str = True
                self.str_is_key = True
                return True
            return False
        if e == 'colon':
            if ch == ':':
                self.expect = 'value'
                return True
            return False
        if e == 'key_end':
            if ch == ',':
                self.expect = 'key'
                return True
            if ch == '}':
                self.stack.pop()
                self._after_value()
                return True
            return False
        if e == 'arr_end':
            if ch == ',':
                self.expect = 'value'
                return True
            if ch == ']':
                self.stack.pop()
                self._after_value()
                return True
            return False
        return False

    def _start_value(self, ch: str) -> bool:
        if ch == ']' and self.stack and self.stack[-1][0] == 'arr':
            # only valid for an empty array
            if not self.stack[-1][1]:
                self.stack.pop()
                self._after_value()
                return True
            return False
        # Any other value marks the enclosing array as non-empty.
        if self.stack and self.stack[-1][0] == 'arr':
            self.stack[-1][1] = True
        if ch == '{':
            self.stack.append(['obj', False])
            self.expect = 'key'
            return True
        if ch == '[':
            self.stack.append(['arr', False])
            self.expect = 'value'
            return True
        if ch == '"':
            self.in_str = True
            self.str_is_key = False
            return True
        if ch in 'tfn':
            self.lit = _LiteralState(ch)
            return True
        if ch == '-' or ch.isdigit():
            self.num = _NumberState()
            return self.num.step(ch)
        return False

    def _after_string(self):
        if self.str_is_key:
            self.expect = 'colon'
        else:
            self._after_value()

    def _after_value(self):
        if not self.stack:
            self.expect = 'done'
            self.complete = True
        elif self.stack[-1][0] == 'obj':
            self.expect = 'key_end'
        else:
            self.expect = 'arr_end'

    def can_accept(self) -> bool:
        if self.complete:
            return True
        if self.stack or self.in_str:
            return False
        if self.num is not None:
            return self.num.is_complete()
        if self.lit is not None:
            return self.lit.done
        return False

    def state_key(self):
        num = None if self.num is None else (
            self.num.phase, self.num.saw_digit, self.num.allow_sign,
            self.num.done, self.num.frac_digit, self.num.exp_digit)
        lit = None if self.lit is None else (self.lit.target, self.lit.pos, self.lit.done)
        return (tuple(tuple(e) for e in self.stack), self.expect, self.in_str,
                self.esc, self.str_is_key, num, lit, self.complete)


# --------------------------------------------------------------------------- #
# Constraint classes
# --------------------------------------------------------------------------- #
class Constraint:
    """Base class for a per-sequence sampling constraint."""

    def apply(self, logits: torch.Tensor, seq, vocab_offset: int,
              vocab_shard: int, vocab_size: int):
        raise NotImplementedError

    def advance(self, seq, token_id: int):
        """Advance internal state after ``token_id`` is appended to the sequence."""

    def is_complete(self, seq) -> bool:
        return False


class LogitBiasConstraint(Constraint):
    def __init__(self, bias: dict):
        self.bias = {int(k): float(v) for k, v in bias.items()}

    def apply(self, logits, seq, vocab_offset, vocab_shard, vocab_size):
        for tid, b in self.bias.items():
            local = tid - vocab_offset
            if 0 <= local < vocab_shard:
                logits[local] += b


class TokenMaskConstraint(Constraint):
    def __init__(self, allow_only: set | None = None, forbidden: set | None = None):
        self.allow_only = frozenset(allow_only) if allow_only else None
        self.forbidden = frozenset(forbidden) if forbidden else None

    def apply(self, logits, seq, vocab_offset, vocab_shard, vocab_size):
        mask = torch.ones(vocab_shard, dtype=torch.bool, device=logits.device)
        if self.allow_only is not None:
            mask.fill_(False)
            local = [t - vocab_offset for t in self.allow_only
                     if 0 <= t - vocab_offset < vocab_shard]
            if local:
                mask[torch.tensor(local, dtype=torch.long, device=logits.device)] = True
        elif self.forbidden is not None:
            for t in self.forbidden:
                local = t - vocab_offset
                if 0 <= local < vocab_shard:
                    mask[local] = False
        logits.masked_fill_(~mask, float('-inf'))


class ChoiceConstraint(Constraint):
    def __init__(self, choices: list[str], tokenizer):
        self.tokenizer = tokenizer
        self.root: dict = {}
        for choice in choices:
            ids = tokenizer.encode(choice, add_special_tokens=False)
            node = self.root
            for tid in ids:
                node = node.setdefault(tid, {})
            node['<end>'] = True

    def _state(self, seq) -> dict:
        if seq.constraint_state is None:
            seq.constraint_state = self.root
        return seq.constraint_state

    def apply(self, logits, seq, vocab_offset, vocab_shard, vocab_size):
        node = self._state(seq)
        allowed = [t for t in node if t != '<end>']
        mask = torch.zeros(vocab_shard, dtype=torch.bool, device=logits.device)
        local = [t - vocab_offset for t in allowed
                 if 0 <= t - vocab_offset < vocab_shard]
        if local:
            mask[torch.tensor(local, dtype=torch.long, device=logits.device)] = True
        logits.masked_fill_(~mask, float('-inf'))

    def advance(self, seq, token_id):
        node = self._state(seq)
        if token_id in node:
            seq.constraint_state = node[token_id]
        else:
            seq.constraint_state = {'<dead>': True}

    def is_complete(self, seq) -> bool:
        return '<end>' in self._state(seq)


class _GrammarConstraint(Constraint):
    """Shared machinery for JSON / regex constraints."""

    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_index = _get_vocab_index(tokenizer)
        self.vocab_size = vocab_size
        self._mask_cache: dict = {}

    def _grammar(self, seq):
        if seq.constraint_state is None:
            seq.constraint_state = self._new_grammar()
        return seq.constraint_state

    def _new_grammar(self):
        raise NotImplementedError

    def _apply_grammar(self, logits, grammar, vocab_offset, vocab_shard, vocab_size):
        key = (grammar.state_key(), vocab_size)
        full = self._mask_cache.get(key)
        if full is None:
            allowed = self.vocab_index.collect_allowed(grammar)
            full = torch.zeros(vocab_size, dtype=torch.bool)
            for tid in allowed:
                if 0 <= tid < vocab_size:
                    full[tid] = True
            self._mask_cache[key] = full
        local = full[vocab_offset:vocab_offset + vocab_shard]
        logits.masked_fill_(~local.to(logits.device), float('-inf'))

    def advance(self, seq, token_id):
        g = self._grammar(seq)
        s = self.tokenizer.decode([token_id], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        for ch in s:
            g.step(ch)


class JsonConstraint(_GrammarConstraint):
    def __init__(self, tokenizer, vocab_size: int, schema=None):
        super().__init__(tokenizer, vocab_size)
        self.schema = schema

    def _new_grammar(self):
        return JsonGrammar()

    def apply(self, logits, seq, vocab_offset, vocab_shard, vocab_size):
        self._apply_grammar(logits, self._grammar(seq), vocab_offset, vocab_shard,
                            vocab_size)

    def is_complete(self, seq) -> bool:
        return self._grammar(seq).can_accept()


class RegexConstraint(_GrammarConstraint):
    def __init__(self, pattern: str, tokenizer, vocab_size: int):
        super().__init__(tokenizer, vocab_size)
        self.pattern = pattern
        self.nfa_start = compile_regex(pattern)
        # quick sanity check that the regex itself is usable
        try:
            _re.compile(pattern)
        except _re.error:
            pass

    def _new_grammar(self):
        return RegexState(self.nfa_start)

    def apply(self, logits, seq, vocab_offset, vocab_shard, vocab_size):
        self._apply_grammar(logits, self._grammar(seq), vocab_offset, vocab_shard,
                            vocab_size)

    def is_complete(self, seq) -> bool:
        return self._grammar(seq).can_accept()


class CompositeConstraint(Constraint):
    def __init__(self, constraints: list[Constraint]):
        self.constraints = constraints

    def apply(self, logits, seq, vocab_offset, vocab_shard, vocab_size):
        for c in self.constraints:
            c.apply(logits, seq, vocab_offset, vocab_shard, vocab_size)

    def advance(self, seq, token_id):
        for c in self.constraints:
            c.advance(seq, token_id)

    def is_complete(self, seq) -> bool:
        return any(c.is_complete(seq) for c in self.constraints)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def make_constraint(params, tokenizer) -> Constraint | None:
    """Build a constraint from ``SamplingParams``, or ``None`` if unconstrained."""
    parts: list[Constraint] = []

    if getattr(params, 'logit_bias', None):
        parts.append(LogitBiasConstraint(params.logit_bias))
    if getattr(params, 'allowed_token_ids', None):
        parts.append(TokenMaskConstraint(allow_only=set(params.allowed_token_ids)))
    if getattr(params, 'bad_token_ids', None):
        parts.append(TokenMaskConstraint(forbidden=set(params.bad_token_ids)))

    vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer)) or len(tokenizer)

    if getattr(params, 'guided_choice', None):
        parts.append(ChoiceConstraint(list(params.guided_choice), tokenizer))
    if getattr(params, 'guided_json', None) is not None:
        parts.append(JsonConstraint(tokenizer, vocab_size,
                                    schema=params.guided_json))
    if getattr(params, 'guided_regex', None):
        parts.append(RegexConstraint(params.guided_regex, tokenizer, vocab_size))

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return CompositeConstraint(parts)


def seqs_have_constraints(seqs) -> bool:
    return any(getattr(s, 'constraint', None) is not None for s in seqs)


def advance_constraint(seq, token_id: int):
    c = getattr(seq, 'constraint', None)
    if c is not None:
        c.advance(seq, token_id)


def constraint_complete(seq) -> bool:
    c = getattr(seq, 'constraint', None)
    return c is not None and c.is_complete(seq)


__all__ = [
    'Constraint', 'make_constraint', 'seqs_have_constraints',
    'advance_constraint', 'constraint_complete',
]
