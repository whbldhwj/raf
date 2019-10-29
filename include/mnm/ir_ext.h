#pragma once

#include <mnm/ir.h>

/****** mnm::ir::Module ******/
namespace mnm {
namespace ir {

class ModuleNode : public Node {
 public:
  Map<GlobalVar, Function> functions;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("functions", &functions);
  }

  void Add(const GlobalVar& var, const Function& func);

  Function Lookup(const GlobalVar& var) const;

 public:
  static constexpr const char* _type_key = "mnm.ir.Module";
  MNM_DEF_NODE_TYPE_INFO(ModuleNode, Node);
};

class Module : public NodeRef {
 public:
  static Module make(Map<GlobalVar, Function> functions);
  MNM_DEF_NODE_REF_METHODS(Module, NodeRef, ModuleNode);
};

}  // namespace ir
}  // namespace mnm

/****** mnm::ir::Module ******/
namespace mnm {
namespace ir {

using RelayConstantNode = tvm::relay::ConstantNode;
using RelayConstant = tvm::relay::Constant;

class ConstantNode : public RelayConstantNode {
 public:
  NodeRef value{nullptr};
};
using Constant = tvm::relay::Constant;

}  // namespace ir
}  // namespace mnm
