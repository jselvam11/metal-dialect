//===- MetalTypes.td - Metal dialect types ---------------*- tablegen -*-===//
//
// This source file is part of the Metal open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_TYPES
#define METAL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "metal/IR/MetalDialect.td"

//===----------------------------------------------------------------------===//
// Metal type definitions
//===----------------------------------------------------------------------===//

class Metal_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Metal_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Metal_MemRefType : Metal_Type<"MetalMemRef", "memref"> {
    let summary = "Metal memref type";
    let parameters = (ins "Type":$type, "unsigned":$size);
    let hasCustomAssemblyFormat = 1;
}

def Metal_PtrType : Metal_Type<"MetalPtr", "ptr"> {
  let summary = "Metal pointer type";
  let parameters = (ins "Type":$pointeeType, "uint32_t":$addressSpace);
  let hasCustomAssemblyFormat = 1;
}

def Metal_BlockedLayout : AttrDef<Metal_Dialect, "BlockedLayout"> {
  let summary = "Represents blocked memory layout for tensors";
  let parameters = (ins
    "ArrayRef<int64_t>":$sizePerThread,
    "ArrayRef<int64_t>":$threadsPerWarp,
    "ArrayRef<int64_t>":$warpsPerCTA,
    "ArrayRef<int64_t>":$order
  );
  let mnemonic = "blocked_layout";
}

def Metal_TensorType : Metal_Type<"MetalTensor", "tensor"> {
  let summary = "Metal tensor type";
  let parameters = (ins 
    "Type":$elementType,
    "ArrayRef<int64_t>":$shape,
    "Attribute":$layout
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // METAL_TYPES
