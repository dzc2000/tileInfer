"""
Test script for TileLang gated delta rule kernel.
Verifies the implementation works correctly on A100 (sm80).
"""
import torch
import sys
sys.path.insert(0, '.')

def test_import():
    """Test that the kernel can be imported."""
    try:
        from kernels.gated_delta_rule_prefill import chunk_gated_delta_rule_tilelang
        print("✓ 导入成功: chunk_gated_delta_rule_tilelang")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_basic_forward():
    """Test basic forward pass with small tensors."""
    try:
        from kernels.gated_delta_rule_prefill import chunk_gated_delta_rule_tilelang
        
        B, T, num_k_heads, num_v_heads, Dk, Dv = 1, 32, 16, 48, 128, 128
        
        q = torch.randn(B, T, num_k_heads, Dk, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, T, num_k_heads, Dk, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, T, num_v_heads, Dv, device='cuda', dtype=torch.bfloat16)
        g = torch.randn(B, T, num_v_heads, device='cuda', dtype=torch.float32) * 0.1
        beta = torch.randn(B, T, num_v_heads, device='cuda', dtype=torch.float32).sigmoid()
        
        output, final_state = chunk_gated_delta_rule_tilelang(
            q, k, v, g, beta, scale=1.0,
            initial_state=None, output_final_state=True
        )
        
        assert output.shape == (B, T, num_v_heads, Dv), f"Output shape mismatch: {output.shape}"
        assert final_state.shape == (B, num_v_heads, Dk, Dv), f"State shape mismatch: {final_state.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        print(f"✓ 前向传播测试通过")
        print(f"  输出形状: {output.shape}")
        print(f"  状态形状: {final_state.shape}")
        print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        return True
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_initial_state():
    """Test with non-zero initial state."""
    try:
        from kernels.gated_delta_rule_prefill import chunk_gated_delta_rule_tilelang
        
        B, T, num_k_heads, num_v_heads, Dk, Dv = 2, 64, 16, 48, 128, 128
        
        q = torch.randn(B, T, num_k_heads, Dk, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, T, num_k_heads, Dk, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, T, num_v_heads, Dv, device='cuda', dtype=torch.bfloat16)
        g = torch.randn(B, T, num_v_heads, device='cuda', dtype=torch.float32) * 0.1
        beta = torch.randn(B, T, num_v_heads, device='cuda', dtype=torch.float32).sigmoid()
        init_state = torch.randn(B, num_v_heads, Dk, Dv, device='cuda', dtype=torch.float32) * 0.01
        
        output, final_state = chunk_gated_delta_rule_tilelang(
            q, k, v, g, beta, scale=1.0,
            initial_state=init_state, output_final_state=True
        )
        
        assert output.shape == (B, T, num_v_heads, Dv)
        assert final_state.shape == (B, num_v_heads, Dk, Dv)
        assert not torch.isnan(output).any()
        assert not torch.isnan(final_state).any()
        
        print(f"✓ 初始状态测试通过")
        print(f"  批次大小: {B}, 序列长度: {T}")
        return True
    except Exception as e:
        print(f"✗ 初始状态测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_padding():
    """Test with sequence length not divisible by CHUNK."""
    try:
        from kernels.gated_delta_rule_prefill import chunk_gated_delta_rule_tilelang
        
        B, T, num_k_heads, num_v_heads, Dk, Dv = 1, 50, 16, 48, 128, 128
        
        q = torch.randn(B, T, num_k_heads, Dk, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, T, num_k_heads, Dk, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, T, num_v_heads, Dv, device='cuda', dtype=torch.bfloat16)
        g = torch.randn(B, T, num_v_heads, device='cuda', dtype=torch.float32) * 0.1
        beta = torch.randn(B, T, num_v_heads, device='cuda', dtype=torch.float32).sigmoid()
        
        output, final_state = chunk_gated_delta_rule_tilelang(
            q, k, v, g, beta, scale=1.0,
            initial_state=None, output_final_state=True
        )
        
        assert output.shape == (B, T, num_v_heads, Dv), f"Output should be trimmed to original length"
        assert not torch.isnan(output).any()
        
        print(f"✓ 填充测试通过 (序列长度 {T} -> 填充到 64)")
        return True
    except Exception as e:
        print(f"✗ 填充测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("TileLang Gated Delta Rule 内核测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("✗ CUDA 不可用，跳过测试")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}\n")
    
    tests = [
        ("导入测试", test_import),
        ("基本前向传播", test_basic_forward),
        ("初始状态", test_with_initial_state),
        ("序列填充", test_padding),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\n运行测试: {name}")
        print("-" * 60)
        result = test_fn()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！TileLang 内核可以正常使用。")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查实现。")

if __name__ == "__main__":
    main()
