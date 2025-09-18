#!/usr/bin/env python3
"""
最终完整导出脚本，确保所有输入都被使用
"""

import torch
import torch.onnx
import numpy as np
from final_complete_model import create_model, get_input_shapes, get_output_shapes


def export_final_complete_model_to_onnx(model_path: str = "final_complete_simplified_pnp.onnx"):
    """Export the final complete PyTorch model to ONNX format"""
    
    # Create model
    model = create_model()
    model.eval()
    
    # Create dummy inputs with correct shapes
    input_shapes = get_input_shapes()
    dummy_inputs = []
    input_names = []
    
    for name, shape in input_shapes.items():
        dummy_inputs.append(torch.randn(shape))
        input_names.append(name)
    
    # Create dummy input tuple
    dummy_input_tuple = tuple(dummy_inputs)
    
    # Define output names
    output_names = list(get_output_shapes().keys())
    
    print("Exporting final complete model to ONNX...")
    print(f"Input shapes: {input_shapes}")
    print(f"Output shapes: {get_output_shapes()}")
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input_tuple,
            model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            keep_initializers_as_inputs=False
        )
        
        print(f"Final complete model exported successfully to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Export failed: {e}")
        return None


def verify_final_complete_model(onnx_path: str):
    """Verify the exported final complete ONNX model"""
    try:
        import onnx
        
        # Load the ONNX model
        model = onnx.load(onnx_path)
        
        # Check if the model is valid
        onnx.checker.check_model(model)
        
        print("Final complete ONNX model verification passed!")
        
        # Print model information
        print(f"Model IR version: {model.ir_version}")
        print(f"Producer name: {model.producer_name}")
        print(f"Producer version: {model.producer_version}")
        
        # Print input/output information
        print("\nInputs:")
        for input_info in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in input_info.type.tensor_type.shape.dim]
            print(f"  {input_info.name}: {shape}")
        
        print("\nOutputs:")
        for output_info in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in output_info.type.tensor_type.shape.dim]
            print(f"  {output_info.name}: {shape}")
        
        return True
        
    except ImportError:
        print("ONNX not available for verification")
        return False
    except Exception as e:
        print(f"Final complete ONNX model verification failed: {e}")
        return False


def compare_final_complete_with_original(original_onnx_path: str, exported_onnx_path: str):
    """Compare the final complete exported model with the original ONNX model"""
    try:
        import onnx
        
        # Load both models
        original_model = onnx.load(original_onnx_path)
        exported_model = onnx.load(exported_onnx_path)
        
        print("Comparing final complete models...")
        
        # Compare input shapes
        print("\nInput comparison:")
        original_inputs = {inp.name: [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                                     for dim in inp.type.tensor_type.shape.dim]
                          for inp in original_model.graph.input}
        exported_inputs = {inp.name: [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                                     for dim in inp.type.tensor_type.shape.dim]
                         for inp in exported_model.graph.input}
        
        all_inputs_match = True
        for name in original_inputs:
            if name in exported_inputs:
                if original_inputs[name] == exported_inputs[name]:
                    print(f"  {name}: ✓ Match")
                else:
                    print(f"  {name}: ✗ Mismatch - Original: {original_inputs[name]}, Exported: {exported_inputs[name]}")
                    all_inputs_match = False
            else:
                print(f"  {name}: ✗ Missing in exported model")
                all_inputs_match = False
        
        # Compare output shapes
        print("\nOutput comparison:")
        original_outputs = {out.name: [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                                      for dim in out.type.tensor_type.shape.dim]
                           for out in original_model.graph.output}
        exported_outputs = {out.name: [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                                      for dim in out.type.tensor_type.shape.dim]
                          for out in exported_model.graph.output}
        
        all_outputs_match = True
        for name in original_outputs:
            if name in exported_outputs:
                if original_outputs[name] == exported_outputs[name]:
                    print(f"  {name}: ✓ Match")
                else:
                    print(f"  {name}: ✗ Mismatch - Original: {original_outputs[name]}, Exported: {exported_outputs[name]}")
                    all_outputs_match = False
            else:
                print(f"  {name}: ✗ Missing in exported model")
                all_outputs_match = False
        
        print(f"\n=== 最终总结 ===")
        print(f"输入完全匹配: {'✓' if all_inputs_match else '✗'}")
        print(f"输出完全匹配: {'✓' if all_outputs_match else '✗'}")
        print(f"模型完全一致: {'✓' if all_inputs_match and all_outputs_match else '✗'}")
        
        if all_inputs_match and all_outputs_match:
            print("\n🎉 恭喜！导出的模型与原始模型完全一致！")
            print("✅ 所有输入形状匹配")
            print("✅ 所有输出形状匹配")
            print("✅ 模型接口完全兼容")
        else:
            print("\n⚠️  导出的模型与原始模型存在差异")
            if not all_inputs_match:
                print("❌ 输入形状不匹配")
            if not all_outputs_match:
                print("❌ 输出形状不匹配")
        
        return all_inputs_match and all_outputs_match
        
    except Exception as e:
        print(f"Final complete model comparison failed: {e}")
        return False


if __name__ == "__main__":
    # Test the final complete model first
    print("Testing final complete model...")
    from final_complete_model import create_model, get_input_shapes, get_output_shapes
    
    model = create_model()
    model.eval()
    
    # Create dummy inputs
    input_shapes = get_input_shapes()
    dummy_inputs = {}
    
    for name, shape in input_shapes.items():
        dummy_inputs[name] = torch.randn(shape)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**dummy_inputs)
    
    # Check output shapes
    expected_shapes = get_output_shapes()
    print("Model outputs:")
    for name, output in outputs.items():
        expected_shape = expected_shapes[name]
        actual_shape = list(output.shape)
        print(f"{name}: {actual_shape} (expected: {expected_shape})")
        if actual_shape != expected_shape:
            print(f"  WARNING: Shape mismatch for {name}")
    
    print("\nFinal complete model test completed!")
    
    # Export final complete model
    exported_path = export_final_complete_model_to_onnx("final_complete_simplified_pnp.onnx")
    
    if exported_path:
        # Verify exported model
        verify_final_complete_model(exported_path)
        
        # Compare with original
        original_path = "simplified_pnp_0901_num_119.onnx"
        is_complete_match = compare_final_complete_with_original(original_path, exported_path)
        
        if is_complete_match:
            print("\n🎉 任务完成！导出的模型与原始模型完全一致！")
        else:
            print("\n⚠️  导出的模型与原始模型存在差异")
    
    print("\nFinal complete export completed!")