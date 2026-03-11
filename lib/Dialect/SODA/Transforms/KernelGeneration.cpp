//===- KernelGeneration.cpp - Implementation of SODA kernel generation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SODA dialect kernel genration for Bambu pass.
//
// It pulls the kernel code into a completely isolated, clean of soda operations
// mlir file that can be further optimized by regular mlir-opt and subsequently
// lowered into llvm ir.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

#include "PassDetail.h"
#include "soda/Dialect/SODA/Passes.h"
#include "soda/Dialect/SODA/SODADialect.h"
#include "soda/Dialect/SODA/Utils.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

class SodaKernelGenerationPass
    : public SodaKernelGenerationBase<SodaKernelGenerationPass> {
public:
  void runOnOperation() override;
};

} // namespace

void SodaKernelGenerationPass::runOnOperation() {

  // Steps:
  // 1 Transfer SODAModuleOp region to its parent mlir::ModuleOp
  // 2 Delete any code unrelated to the kernel (if includeHost is false)
  // 3 Walk through the module and change soda.module terminator
  // 4 Walk through the module and change soda.func to regular func
  // 5 Walk through the module and change soda.return
  // 6 Set attribute marking that we modified this mlir:ModuleOp for Bambu
  // target

  Operation *op = getOperation();

  if (!isa<ModuleOp>(op)) {
    return signalPassFailure();
  }

  ModuleOp mop = dyn_cast<ModuleOp>(op);
  if (!mop) {
    return signalPassFailure();
  }

  OpBuilder builder(mop.getContext());

  // Collect all SODAModuleOps first to avoid iterator invalidation during erase
  SmallVector<soda::SODAModuleOp, 4> sodaModules;
  mop.walk([&](soda::SODAModuleOp sodaOp) {
    sodaModules.push_back(sodaOp);
  });

  if (sodaModules.empty()) {
    return signalPassFailure();
  }

  bool modified = true;
  Block &mainBlock = mop.getRegion().front();
  for (auto sodaOp : sodaModules) {

    // Rename all symbols in the SODAModuleOp to avoid collisions at top level
    SymbolTable symbolTable(sodaOp);
    for (auto &op :
         llvm::make_early_inc_range(sodaOp.getRegion().front().getOperations())) {
      if (isa<soda::ModuleEndOp>(op))
        continue;

      if (auto symbol = dyn_cast<SymbolOpInterface>(op)) {
        std::string newNameStr =
            (Twine(sodaOp.getName()) + "_" + Twine(symbol.getName())).str();
        StringAttr newName = builder.getStringAttr(newNameStr);

        if (failed(SymbolTable::replaceAllSymbolUses(symbol, newName, sodaOp))) {
          return signalPassFailure();
        }
        symbol.setName(newName);
      }
    }

    Region &sodaRegion = sodaOp.getRegion();
    for (auto &op :
         llvm::make_early_inc_range(sodaRegion.front().getOperations())) {
      if (isa<soda::ModuleEndOp>(op))
        continue;
      // Move kernel function to the top-level module
      op.moveBefore(&mainBlock, mainBlock.end());
    }
    sodaOp.erase();
  }

  // If includeHost is true, we must convert soda.launch_func to func.call
  if (this->includeHost) {
    mop.walk([&](soda::LaunchFuncOp launchOp) {
      std::string newName = (Twine(launchOp.getKernelModuleName()) + "_" +
                             Twine(launchOp.getKernelName()))
                                .str();
      OpBuilder builder(launchOp);
      builder.create<func::CallOp>(launchOp.getLoc(), newName, TypeRange{},
                                   launchOp.getKernelOperands());
      launchOp.erase();
    });
  }

  // If includeHost is false, we delete everything that was in the original module
  // except the newly moved kernels.
  if (!this->includeHost) {
    // Move kernels to a temporary place or just delete everything else
    // Actually, the previous loop already moved them to the end of mainBlock.
    // We want to delete the operations that were there *before* our kernels.
    
    // Let's re-think: the kernels are at the end. 
    // We can delete ops from the beginning until we hit a SODAFuncOp.
    while (!mainBlock.empty() && !isa<soda::SODAFuncOp>(mainBlock.front())) {
      mainBlock.front().erase();
    }
  }

  mop.walk([](soda::ModuleEndOp endOp) { endOp.erase(); });

  mop.walk([this](soda::SODAFuncOp funcOp) {
    OpBuilder replacer(funcOp);

    func::FuncOp dstFunc = replacer.create<func::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());

    dstFunc.getRegion().takeBody(funcOp.getBody());
    funcOp.erase();

    // Set all memref arguments to noalias
    // TODO (NICO): Create analysis on the outliner, only carry decisions here

    if (!(this->noAliasAnalysis)) {
      int index = 0;
      for (BlockArgument argument : dstFunc.getArguments()) {
        if (isa<MemRefType>(argument.getType())) {
          dstFunc.setArgAttr(index, LLVMDialect::getNoAliasAttrName(),
                             UnitAttr::get(dstFunc.getContext()));
        }
        index++;
      }
    }
  });

  mop.walk([](soda::ReturnOp returnOp) {
    OpBuilder replacer(returnOp);
    replacer.create<mlir::func::ReturnOp>(returnOp.getLoc());
    returnOp.erase();
  });

  if (modified)
    mop->setAttr("soda.bambu.container_module", UnitAttr::get(&getContext()));
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::soda::createSodaKernelGenerationPass() {
  return std::make_unique<SodaKernelGenerationPass>();
}