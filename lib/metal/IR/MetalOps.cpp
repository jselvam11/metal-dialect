//===--- MetalOps.cpp - Metal dialect ops ---------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/IR/MetalOps.h"
#include "metal/IR/MetalDialect.h"
#include "metal/IR/MetalOpsEnums.cpp.inc"
#include "metal/IR/MetalTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir::metal;
using namespace mlir;

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

void ModuleOp::build(OpBuilder &builder, OperationState &result) {
  ensureTerminator(*result.addRegion(), builder, result.location);
}

void ModuleOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printRegion(getRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

mlir::ParseResult ModuleOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, std::nullopt))
    return mlir::failure();
  ModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

void KernelOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     llvm::SmallVectorImpl<Type> &buffers,
                     llvm::SmallVectorImpl<bool> &isAddressSpaceDevice) {
  result.addTypes(std::nullopt);
  result.addAttribute("name", builder.getStringAttr(name));
  result.addAttribute("address_space_device",
                      builder.getBoolArrayAttr(isAddressSpaceDevice));
  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  auto block = builder.createBlock(bodyRegion);
  for (auto type : buffers) {
    auto memref = MetalMemRefType::get(builder.getContext(), type, 0);
    block->addArguments(memref, result.location);
  }
}

mlir::Block &KernelOp::getEntryBlock() { return getRegion().front(); }

llvm::LogicalResult KernelOp::verify() {
  auto index = -1;
  for (auto it : llvm::enumerate(getBuffers())) {
    auto memRef =
        llvm::dyn_cast<mlir::metal::MetalMemRefType>(it.value().getType());
    if (!memRef) {
      index = it.index();
      break;
    }

    auto type = memRef.getType();
    if (type.isF16() || type.isF32() || type.isIndex())
      continue;
    if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type)) {
      switch (intTy.getWidth()) {
      case 1:
      case 8:
      case 16:
      case 32:
      case 64:
        continue;
      }
    }

    index = it.index();
    break;
  }
  if (index != -1)
    return emitOpError() << "type #" << index << " must be compatible type";
  else
    return mlir::success();
}

mlir::Value KernelOp::getBuffer(uint32_t index) {
  return getBodyRegion().getBlocks().begin()->getArgument(index);
}

mlir::MutableArrayRef<mlir::BlockArgument> KernelOp::getBuffers() {
  return getBodyRegion().getBlocks().begin()->getArguments();
}

void KernelOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getName();
  printer << " address_space_device ";
  printer.printAttribute(getAddressSpaceDevice());
  printer << " ";
  printer.printRegion(getRegion(),
                      /*printEntryBlockArgs=*/true,
                      /*printBlockTerminators=*/true);
}

mlir::ParseResult KernelOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  llvm::StringRef name;
  mlir::Region *body = result.addRegion();
  mlir::Attribute value;
  if (parser.parseKeyword(&name) ||
      parser.parseKeyword("address_space_device") ||
      parser.parseAttribute(value, "address_space_device", result.attributes) ||
      parser.parseRegion(*body, std::nullopt))
    return mlir::failure();

  result.addAttribute("name", parser.getBuilder().getStringAttr(name));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       TypedAttr attr) {
  ConstantOp::build(builder, state, attr.getType(), attr);
}

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult AllocaOp::verify() {
  if (llvm::dyn_cast<MetalMemRefType>(getResult().getType()).getSize() == 0)
    return emitOpError() << "memRef size cannot be 0";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Check Index
//===----------------------------------------------------------------------===//

static llvm::LogicalResult
checkIndex(mlir::Operation *op, MetalMemRefType memRef, mlir::Value index) {
  if (auto constantOp = index.getDefiningOp<ConstantOp>()) {
    auto attr = llvm::dyn_cast<mlir::IntegerAttr>(constantOp.getValue());
    uint64_t index = attr.getUInt();
    uint32_t size = memRef.getSize();
    if (size > 0 && index >= size)
      return op->emitOpError()
             << "index " << index << " is past the end of the memRef "
             << "(which contains " << size << " elements)";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult StoreOp::verify() {
  auto memRef = llvm::dyn_cast<MetalMemRefType>(getMemref().getType());
  auto valueType = getValue().getType();
  auto memRefType = memRef.getType();
  if (memRefType != valueType)
    return emitOpError() << "requires value's type (" << valueType
                         << ") to match memref type (" << memRefType << ")";
  return checkIndex(*this, memRef, getIndex());
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

void GetElementOp::build(OpBuilder &builder, OperationState &result,
                         Value memref, Value index) {
  result.addOperands(memref);
  result.addOperands(index);
  auto type = llvm::cast<MetalMemRefType>(memref.getType()).getType();
  result.types.push_back(type);
};

llvm::LogicalResult GetElementOp::verify() {
  auto memRef = llvm::dyn_cast<MetalMemRefType>(getMemref().getType());
  auto resultType = getResult().getType();
  auto memRefType = memRef.getType();
  if (memRefType != resultType)
    return emitOpError() << "requires memref type (" << memRefType
                         << ") to match return type (" << resultType << ")";

  return checkIndex(*this, memRef, getIndex());
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

void ThreadIdOp::build(OpBuilder &builder, OperationState &result,
                       StringRef dimension) {
  result.addAttribute("dimension", builder.getStringAttr(dimension));
  result.addTypes(builder.getIntegerType(32, false));
};

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ThreadIdOp::verify() {
  auto dim = getDimension();
  if (dim != "x" && dim != "y" && dim != "z")
    return emitOpError() << "requires dimension to be `x` or `y` or `z`, "
                         << "found `" << dim << "`";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnaryExpOp
//===----------------------------------------------------------------------===//

void UnaryExpOp::build(OpBuilder &builder, OperationState &result,
                       UnaryExpOperator unaryOperator, Value argument) {
  result.addTypes(argument.getType());
  auto attr = builder.getI64IntegerAttr(static_cast<int64_t>(unaryOperator));
  result.addAttribute("unaryOperator", attr);
  result.addOperands(argument);
}

llvm::LogicalResult UnaryExpOp::verify() {
  auto argType = getType();
  auto resultType = getResult().getType();
  if (argType != resultType)
    return emitOpError() << "result type mismatch";

  using OP = mlir::metal::UnaryExpOperator;
  switch (getUnaryOperator()) {
  case OP::notOp:
    if (!argType.isInteger(1))
      return emitOpError() << "argument type must be i1";
    break;
  case OP::minusOp:
    if (argType.isInteger(1) ||
        (!argType.isSignedInteger() && !argType.isF16() && !argType.isF32()))
      return emitOpError() << "argument type must be signed integer or float";
    break;
  }
  return mlir::success();
}

mlir::OpFoldResult UnaryExpOp::fold(FoldAdaptor adaptor) {
  auto constant =
      dyn_cast<mlir::metal::ConstantOp>(getArgument().getDefiningOp());
  if (!constant)
    return nullptr;

  mlir::OpBuilder builder{constant.getContext()};

  switch (getUnaryOperator()) {
  case mlir::metal::UnaryExpOperator::notOp: {
    auto value =
        llvm::dyn_cast<mlir::BoolAttr>(constant.getValueAttr()).getValue();
    return builder.getBoolAttr(!value);
  }
  case mlir::metal::UnaryExpOperator::minusOp: {
    auto attr = constant.getValueAttr();
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
      return builder.getIntegerAttr(attr.getType(), -intAttr.getSInt());
    }
    if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
      return builder.getFloatAttr(attr.getType(),
                                  -floatAttr.getValueAsDouble());
    }
    return nullptr;
  }
  }
}

//===----------------------------------------------------------------------===//
// BinaryExpOp
//===----------------------------------------------------------------------===//

void BinaryExpOp::build(OpBuilder &builder, OperationState &result,
                        BinaryExpOperator binaryOperator, Value lhs,
                        Value rhs) {
  using OP = mlir::metal::BinaryExpOperator;
  switch (binaryOperator) {
  case OP::addOp:
  case OP::subOp:
  case OP::mulOp:
  case OP::divOp:
  case OP::remOp:
    result.addTypes(lhs.getType());
    break;
  case OP::eqOp:
  case OP::neOp:
  case OP::ltOp:
  case OP::leOp:
  case OP::gtOp:
  case OP::geOp:
  case OP::andOp:
  case OP::orOp:
    result.addTypes(builder.getI1Type());
    break;
  }

  auto attr = builder.getI64IntegerAttr(static_cast<int64_t>(binaryOperator));
  result.addAttribute("binaryOperator", attr);
  result.addOperands(lhs);
  result.addOperands(rhs);
}

llvm::LogicalResult BinaryExpOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto resultType = getResult().getType();
  if (lhsType != rhsType)
    return emitOpError() << "arguments type mismatch";

  using OP = mlir::metal::BinaryExpOperator;
  switch (getBinaryOperator()) {
  case OP::addOp:
  case OP::subOp:
  case OP::mulOp:
  case OP::divOp:
  case OP::remOp:
    if (lhsType != resultType)
      return emitOpError() << "result type mismatch";
    break;
  case OP::eqOp:
  case OP::neOp:
  case OP::ltOp:
  case OP::leOp:
  case OP::gtOp:
  case OP::geOp:
  case OP::andOp:
  case OP::orOp:
    if (!resultType.isInteger(1))
      return emitOpError() << "result type mismatch";
    break;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

auto IfOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::Value cond,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> thenBuilder,
    function_ref<void(OpBuilder &, mlir::Location)> elseBuilder) -> void {
  result.addTypes(std::nullopt);
  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);

  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (elseBuilder) {
    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
  }
}

mlir::ParseResult IfOp::parse(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::UnresolvedOperand condition;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(condition) ||
      parser.resolveOperand(condition, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, std::nullopt))
    return mlir::failure();

  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, std::nullopt))
      return mlir::failure();
  }

  return mlir::success();
}

void IfOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getCondition() << " ";
  printer.printRegion(getThenRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  auto &elseRegion = this->getElseRegion();
  if (!elseRegion.empty()) {
    printer << " else ";
    printer.printRegion(elseRegion,
                        /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

auto WhileOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> conditionBuilder,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) -> void {
  result.addTypes(std::nullopt);
  result.addOperands(std::nullopt);

  OpBuilder::InsertionGuard guard(builder);

  Region *conditionRegion = result.addRegion();
  builder.createBlock(conditionRegion);
  conditionBuilder(builder, result.location);

  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  bodyBuilder(builder, result.location);
}

llvm::LogicalResult WhileOp::verify() {
  auto &region = getConditionRegion();

  for (auto it = region.op_begin(); it != region.op_end(); it++) {
    if (!llvm::isa<ConstantOp, StoreOp, GetElementOp, ThreadIdOp, CastOp,
                   UnaryExpOp, BinaryExpOp, YieldWhileOp>(*it))
      return emitOpError() << it->getName()
                           << " op not allowed in the condition region";
  }

  return mlir::success();
}

mlir::ParseResult WhileOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *conditionRegion = result.addRegion();
  mlir::Region *bodyRegion = result.addRegion();
  if (parser.parseKeyword("condition") ||
      parser.parseRegion(*conditionRegion, std::nullopt) ||
      parser.parseKeyword("loop") ||
      parser.parseRegion(*bodyRegion, std::nullopt))
    return mlir::failure();

  return mlir::success();
}

void WhileOp::print(mlir::OpAsmPrinter &printer) {
  printer << " condition ";
  printer.printRegion(getConditionRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  auto &bodyRegion = this->getBodyRegion();
  printer << " loop ";
  printer.printRegion(bodyRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

void DeviceMakeDefaultOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIndexType());
};

void DeviceMakeCommandQueueOp::build(OpBuilder &builder, OperationState &result,
                                     Value device) {
  result.addOperands(device);
  result.addTypes(builder.getIndexType());
};

void DeviceMakeBufferOp::build(OpBuilder &builder, OperationState &result,
                               Value device, Value isStorageModeManaged,
                               Value count, Value sizeType) {
  result.addOperands(device);
  result.addOperands(isStorageModeManaged);
  result.addOperands(count);
  result.addOperands(sizeType);
  result.addTypes(builder.getIndexType());
};

//===----------------------------------------------------------------------===//
// Runtime - Buffer
//===----------------------------------------------------------------------===//

void BufferGetContentsOp::build(OpBuilder &builder, OperationState &result,
                                Value device, Type elementType) {
  result.addOperands(device);
  auto memRefType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
  result.addTypes(memRefType);
};

llvm::LogicalResult BufferGetContentsOp::verify() {
  auto elementType =
      llvm::cast<mlir::MemRefType>(getResult().getType()).getElementType();
  if (isa<mlir::IntegerType>(elementType) || elementType.isF16() ||
      elementType.isF32())
    return mlir::success();

  return emitOpError() << "the buffer has an incompatible type";
}

//===----------------------------------------------------------------------===//
// Runtime - CommandQueue
//===----------------------------------------------------------------------===//

void CommandQueueMakeCommandBufferOp::build(OpBuilder &builder,
                                            OperationState &result,
                                            Value commandQueue,
                                            StringRef functionName, Value width,
                                            Value height, Value depth) {
  result.addAttribute("functionName", builder.getStringAttr(functionName));
  result.addOperands(commandQueue);
  result.addOperands(width);
  result.addOperands(height);
  result.addOperands(depth);
  result.addTypes(builder.getIndexType());
};

void CommandQueueMakeCommandBufferOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getFunctionName() << " ";
  printer << getCommandQueue() << ", ";
  printer << getWidth() << ", ";
  printer << getHeight() << ", ";
  printer << getDepth();
  printer << ": (" << getOperandTypes() << ") -> ";
  printer.printType(getResult().getType());
}

mlir::ParseResult
CommandQueueMakeCommandBufferOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  llvm::StringRef functionName;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  if (parser.parseKeyword(&functionName) ||
      parser.parseOperandList(operands, 4) || parser.parseColon() ||
      parser.parseLParen() || parser.parseTypeList(operandTypes) ||
      parser.parseRParen() || parser.parseArrowTypeList(result.types))
    return mlir::failure();

  result.addAttribute("functionName",
                      parser.getBuilder().getStringAttr(functionName));

  return parser.resolveOperands(operands, operandTypes,
                                parser.getCurrentLocation(), result.operands);
}

//===----------------------------------------------------------------------===//
// GetProgramIdOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult GetProgramIdOp::verify() {
  auto dim = getDimension();
  if (dim != "x" && dim != "y" && dim != "z")
    return emitOpError() << "dimension must be 'x', 'y', or 'z', but got '" << dim << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// MakeRangeOp Implementation
//===----------------------------------------------------------------------===//

// No additional verification needed

//===----------------------------------------------------------------------===//
// SplatOp Implementation
//===----------------------------------------------------------------------===//

void SplatOp::build(OpBuilder &builder, OperationState &result, Value scalar, Type resultType) {
  result.addOperands(scalar);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// AddPtrOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult AddPtrOp::verify() {
  auto pointersType = llvm::dyn_cast<MetalTensorType>(getPointers().getType());
  auto offsetsType = llvm::dyn_cast<MetalTensorType>(getOffsets().getType());
  auto resultType = llvm::dyn_cast<MetalTensorType>(getResult().getType());
  
  if (!pointersType || !offsetsType || !resultType)
    return emitOpError() << "all operands and results must be tensor types";
  
  // Check that shapes match
  if (pointersType.getShape() != offsetsType.getShape() || 
      pointersType.getShape() != resultType.getShape())
    return emitOpError() << "all tensors must have the same shape";
  
  // Check that pointers tensor contains pointers
  auto pointersElementType = llvm::dyn_cast<MetalPtrType>(pointersType.getElementType());
  if (!pointersElementType)
    return emitOpError() << "pointers operand must be a tensor of pointers";
  
  // Check that offsets tensor contains integers
  auto offsetsElementType = offsetsType.getElementType();
  if (!offsetsElementType.isIntOrIndex())
    return emitOpError() << "offsets operand must be a tensor of integers";
  
  // Check that result tensor contains pointers of the same type
  auto resultElementType = llvm::dyn_cast<MetalPtrType>(resultType.getElementType());
  if (!resultElementType)
    return emitOpError() << "result must be a tensor of pointers";
  
  if (resultElementType.getPointeeType() != pointersElementType.getPointeeType())
    return emitOpError() << "result pointer type must match input pointer type";
  
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult TensorLoadOp::verify() {
  auto pointersType = llvm::dyn_cast<MetalTensorType>(getPointers().getType());
  auto maskType = llvm::dyn_cast<MetalTensorType>(getMask().getType());
  auto resultType = llvm::dyn_cast<MetalTensorType>(getResult().getType());
  
  if (!pointersType || !maskType || !resultType)
    return emitOpError() << "all operands and results must be tensor types";
  
  // Check that shapes match
  if (pointersType.getShape() != maskType.getShape() || 
      pointersType.getShape() != resultType.getShape())
    return emitOpError() << "all tensors must have the same shape";
  
  // Check that pointers tensor contains pointers
  auto pointersElementType = llvm::dyn_cast<MetalPtrType>(pointersType.getElementType());
  if (!pointersElementType)
    return emitOpError() << "pointers operand must be a tensor of pointers";
  
  // Check that mask tensor contains booleans
  auto maskElementType = maskType.getElementType();
  if (!maskElementType.isInteger(1))
    return emitOpError() << "mask operand must be a tensor of i1 (boolean) values";
  
  // Check that result element type matches pointee type
  if (resultType.getElementType() != pointersElementType.getPointeeType())
    return emitOpError() << "result element type must match pointer pointee type";
  
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult TensorStoreOp::verify() {
  auto pointersType = llvm::dyn_cast<MetalTensorType>(getPointers().getType());
  auto valuesType = llvm::dyn_cast<MetalTensorType>(getValues().getType());
  auto maskType = llvm::dyn_cast<MetalTensorType>(getMask().getType());
  
  if (!pointersType || !valuesType || !maskType)
    return emitOpError() << "all operands must be tensor types";
  
  // Check that shapes match
  if (pointersType.getShape() != valuesType.getShape() || 
      pointersType.getShape() != maskType.getShape())
    return emitOpError() << "all tensors must have the same shape";
  
  // Check that pointers tensor contains pointers
  auto pointersElementType = llvm::dyn_cast<MetalPtrType>(pointersType.getElementType());
  if (!pointersElementType)
    return emitOpError() << "pointers operand must be a tensor of pointers";
  
  // Check that mask tensor contains booleans
  auto maskElementType = maskType.getElementType();
  if (!maskElementType.isInteger(1))
    return emitOpError() << "mask operand must be a tensor of i1 (boolean) values";
  
  // Check that values element type matches pointee type
  if (valuesType.getElementType() != pointersElementType.getPointeeType())
    return emitOpError() << "values element type must match pointer pointee type";
  
  return success();
}

//===----------------------------------------------------------------------===//
// TensorBinaryOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult TensorBinaryOp::verify() {
  auto lhsType = llvm::dyn_cast<MetalTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<MetalTensorType>(getRhs().getType());
  auto resultType = llvm::dyn_cast<MetalTensorType>(getResult().getType());
  
  if (!lhsType || !rhsType || !resultType)
    return emitOpError() << "all operands and results must be tensor types";
  
  // Check that shapes match
  if (lhsType.getShape() != rhsType.getShape() || 
      lhsType.getShape() != resultType.getShape())
    return emitOpError() << "all tensors must have the same shape";
  
  // Check that element types match or are compatible
  auto lhsElementType = lhsType.getElementType();
  auto rhsElementType = rhsType.getElementType();
  auto resultElementType = resultType.getElementType();
  
  // For comparison operators, result should be i1
  using OP = mlir::metal::BinaryExpOperator;
  switch (getOpr()) {
  case OP::eqOp:
  case OP::neOp:
  case OP::ltOp:
  case OP::leOp:
  case OP::gtOp:
  case OP::geOp:
    if (!resultElementType.isInteger(1))
      return emitOpError() << "comparison operations must return boolean tensors";
    break;
  default:
    // For other operators, result type should match operand types
    if (resultElementType != lhsElementType || resultElementType != rhsElementType)
      return emitOpError() << "element types of operands and result must match for arithmetic operations";
    break;
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// TensorUnaryOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult TensorUnaryOp::verify() {
  auto inputType = llvm::dyn_cast<MetalTensorType>(getInput().getType());
  auto resultType = llvm::dyn_cast<MetalTensorType>(getResult().getType());
  
  if (!inputType || !resultType)
    return emitOpError() << "all operands and results must be tensor types";
  
  // Check that shapes match
  if (inputType.getShape() != resultType.getShape())
    return emitOpError() << "input and result tensors must have the same shape";
  
  // Check correct element types based on operation
  auto inputElementType = inputType.getElementType();
  auto resultElementType = resultType.getElementType();
  
  using OP = mlir::metal::UnaryExpOperator;
  switch (getOpr()) {
  case OP::notOp:
    if (!inputElementType.isInteger(1))
      return emitOpError() << "logical NOT requires boolean input tensor";
    if (!resultElementType.isInteger(1))
      return emitOpError() << "logical NOT must return boolean tensor";
    break;
  case OP::minusOp:
    if (!inputElementType.isIntOrFloat())
      return emitOpError() << "arithmetic negation requires numeric input tensor";
    if (resultElementType != inputElementType)
      return emitOpError() << "arithmetic negation must preserve element type";
    break;
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen's op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "metal/IR/MetalOps.cpp.inc"
