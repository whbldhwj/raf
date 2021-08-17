"""
The visualizer utilities.
"""
# pylint: disable=too-many-arguments, too-many-instance-attributes, missing-function-docstring
# pylint: disable=too-many-public-methods
from collections import defaultdict
import pydot
import tvm
from tvm import relay
from mnm._core.value import FloatValue, IntValue, StringValue
from mnm._ffi.ir.constant import ExtractValue


class DataflowGraphDrawer(tvm.relay.ExprFunctor):
    """
    The dataflow graph drawer.
    """

    def __init__(self, expr, always_draw_exprs=None, graph_label="", draw_atomic_nodes=False,
                 draw_event_nodes=False):
        super().__init__()
        self.expr = expr
        self.always_draw_exprs = always_draw_exprs if always_draw_exprs else []
        self.draw_atomic_nodes = draw_atomic_nodes
        self.draw_event_nodes = draw_event_nodes
        self.graph = pydot.Dot(graph_type='digraph', label=graph_label)

        self.event_expr = {}
        self.current_stream = 0
        self.stream_exprs = defaultdict(list)
        self.stream_wait_events = defaultdict(list)

    def draw(self) -> pydot.Dot:
        self.visit(self.expr)
        return self.graph

    def need_draw(self, e):
        # Always draw some exprs
        if e in self.always_draw_exprs:
            return True
        # Do not draw atomic expr if draw_atomic_nodes is turned off
        if self.is_atomic_expr(e) and not self.draw_atomic_nodes:
            return False
        # Draw all non-atomic expr
        return True

    @staticmethod
    def is_atomic_expr(e):
        return not isinstance(e, (relay.Call, relay.Tuple, relay.TupleGetItem))

    @staticmethod
    def is_scalar_value(value):
        return isinstance(value, (IntValue, FloatValue, StringValue))

    @staticmethod
    def get_fused_op_name(func):
        class FusedOpCalleeVisitor(tvm.relay.ExprVisitor):
            """
            Collect the callee names in the relay function.
            """

            def __init__(self, func):
                super().__init__()
                self.func = func
                self.callees = []

            def get_callee_names(self):
                self.visit(func)
                return self.callees

            def visit_call(self, call):
                op = call.op
                if isinstance(op, tvm.ir.Op):
                    self.callees.append(op.name.split('.')[-1])
                elif isinstance(op, tvm.relay.Function):
                    self.callees.extend(FusedOpCalleeVisitor(op).get_callee_names())
                tvm.relay.ExprVisitor.visit_call(self, call)

        callee_names = FusedOpCalleeVisitor(func).get_callee_names()
        return 'fused_' + '_'.join(callee_names)

    def add_node(self, e, label, event_node=False):
        if self.need_draw(e):
            node_style = {
                'shape': 'box',
                'style': '"rounded,filled"'
            }
            colors = [
                "beige", "azure", "burlywood", "coral1", "darkgoldenrod",
                "darkgreen", "darkorchid2", "firebrick1", "gold1", "antiquewhite", "aquamarine",
                "chartreuse1", "crimson",
            ]
            node_style['fillcolor'] = colors[self.current_stream % len(colors)]
            if e in self.always_draw_exprs:
                node_style['fillcolor'] = 'white'
            if event_node:
                node_style['shape'] = 'diamond'
                node_style['fillcolor'] = 'white'
            node = pydot.Node(name=str(len(self.memo_map)), label=label, **node_style)
            self.stream_exprs[self.current_stream].append(e)
            self.graph.add_node(node)
            self.memo_map[e] = node
        else:
            self.memo_map[e] = None

    def wait_events(self, e):
        event_ids = self.stream_wait_events[self.current_stream]
        if len(event_ids) > 0:
            for event_id in event_ids:
                self.add_edge(self.event_expr[event_id], e, control_edge=True)
            self.stream_wait_events[self.current_stream].clear()

    def add_edge(self, u_expr, v_expr, control_edge=False):
        u_node = self.memo_map[u_expr]
        v_node = self.memo_map[v_expr]
        if u_node and v_node:
            edge_style = {}
            if control_edge:
                edge_style['style'] = 'dashed'
            self.graph.add_edge(pydot.Edge(u_node, v_node, **edge_style))

    def visit_function(self, e):
        attrs = e.attrs
        if attrs and 'Primitive' in attrs and attrs['Primitive'] == 1:
            self.add_node(e, self.get_fused_op_name(e))
            return self.memo_map[e]
        raise NotImplementedError("Does not support a graph with non-primitive functions")

    def visit_let(self, let):
        self.visit(let.value)
        self.memo_map[let.var] = self.memo_map[let.value]
        self.visit(let.body)
        self.memo_map[let] = None
        return self.memo_map[let]

    def visit_call(self, call):
        op = call.op

        # deal with the schedule-related op specially
        schedule_ops = ["mnm.op.set_stream", "mnm.op.add_event", "mnm.op.wait_event"]
        if isinstance(op, tvm.ir.Op) and op.name in schedule_ops:
            if op.name == "mnm.op.set_stream":
                self.current_stream = ExtractValue(call.args[1]).value
                self.memo_map[call] = None
            else:
                if self.draw_event_nodes:
                    event_id = ExtractValue(call.args[0]).value
                    if op.name == "mnm.op.add_event":
                        self.add_node(call, f"Event({event_id})", event_node=True)
                        if len(self.stream_exprs[self.current_stream]) > 1:
                            prev_expr = self.stream_exprs[self.current_stream][-2]
                            self.add_edge(prev_expr, call, control_edge=True)
                        self.wait_events(call)
                        self.event_expr[event_id] = call
                    elif op.name == "mnm.op.wait_event":
                        self.stream_wait_events[self.current_stream].append(event_id)
                        self.memo_map[call] = None
        else:
            self.visit(op)
            if isinstance(op, tvm.ir.Op):
                self.add_node(call, f'Call({op.name.split(".")[-1]})')
            else:
                self.add_node(call, f'Call({self.get_fused_op_name(op)})')
            self.wait_events(call)
            self.add_edge(op, call)
            for arg in call.args:
                self.visit(arg)
                self.add_edge(arg, call)
        return self.memo_map[call]

    def visit_var(self, var):
        last_name = var.name_hint.split('.')[-1]
        self.add_node(var, f"Var({last_name})")
        return self.memo_map[var]

    def visit_tuple(self, tup):
        self.add_node(tup, "Tuple")
        self.wait_events(tup)
        for field in tup.fields:
            self.visit(field)
            self.add_edge(field, tup)
        return self.memo_map[tup]

    def visit_tuple_getitem(self, tup_item):
        self.add_node(tup_item, f"TupleGetItem({tup_item.index})")
        self.wait_events(tup_item)
        self.visit(tup_item.tuple_value)
        self.add_edge(tup_item.tuple_value, tup_item)
        return self.memo_map[tup_item]

    def visit_global_var(self, global_var):
        assert isinstance(global_var, tvm.ir.GlobalVar)
        self.add_node(global_var, f"GlobalVar({global_var.name_hint})")
        return self.memo_map[global_var]

    def visit_op(self, op):
        last_name = op.name.split('.')[-1]
        self.add_node(op, f"Op({last_name})")
        return self.memo_map[op]

    def visit_constant(self, const):
        value = ExtractValue(const)
        if self.is_scalar_value(value):
            label = f'Scalar({str(value)})'
        else:
            label = 'Constant'
        self.add_node(const, label)
        return self.memo_map[const]

    def visit_type(self, typ):
        return typ

    def visit_ref_create(self, _):
        raise NotImplementedError()

    def visit_ref_write(self, _):
        raise NotImplementedError()

    def visit_ref_read(self, _):
        raise NotImplementedError()

    def visit_constructor(self, _):
        raise NotImplementedError()

    def visit_match(self, _):
        raise NotImplementedError()

    def visit_if(self, _):
        raise NotImplementedError()


def draw_dataflow_graph(mod_or_func_or_expr,
                        out_file_name="./graph.png",
                        graph_label='Dataflow Graph',
                        num_inputs=1,
                        draw_atomic_nodes=False,
                        draw_event_nodes=False):
    """
    Draw the dataflow graph of given module, relay function or expression. When a module is given,
    the 'main' function is drawn. The input expr or function can be either GNF, BBNF, or ANF. If
    the given function or expression are scheduled (i.e. after StreamSchedule pass), nodes on
    different CUDA streams would be in different color.

    Parameters
    ----------
    mod_or_func_or_expr : Union[tvm.ir.IRModule, tvm.relay.Function, tvm.ir.RelayExpr]
        The ir module, relay function or expression to be drawn. If a module is given, the main
        function is drawn. We can use draw_dataflow_graph(mod['other_func']) to draw other function
        in the ir module (in this example, 'other_func' is the function's name), if needed.

    out_file_name : str
        The output file name to save the image. Default: './graph.png'.

    graph_label : str
        The graph label to be shown in the image. Default: 'Dataflow Graph'.

    num_inputs : int
        When drawing a function, the first num_inputs of the parameters are always drawn, no matter
        whether draw_atomic_nodes is turned on or off. Default: 1.

    draw_atomic_nodes : bool
        Whether to draw the atomic nodes. We take all expressions other than Call, Tuple and
        TupleGetItem as atomic nodes. Default: False.

    draw_event_nodes : bool
        Whether to draw the event node and control dependency. All data dependency are drawn in
        solid line and the control dependency are drawn in dashed line. Default: False.
    """
    if isinstance(mod_or_func_or_expr, tvm.ir.IRModule):
        expr = mod_or_func_or_expr['main'].body
        always_draw_exprs = mod_or_func_or_expr['main'].params[:num_inputs]
    elif isinstance(mod_or_func_or_expr, tvm.relay.Function):
        expr = mod_or_func_or_expr.body
        always_draw_exprs = mod_or_func_or_expr.params[:num_inputs]
    elif isinstance(mod_or_func_or_expr, tvm.relay.Expr):
        expr = mod_or_func_or_expr
        always_draw_exprs = []
    else:
        raise ValueError("Expect tvm.ir.IRModule, tvm.relay.Function, or tvm.relay.Expr, "
                         f"but {type(mod_or_func_or_expr)} got.")

    drawer = DataflowGraphDrawer(expr,
                                 always_draw_exprs=always_draw_exprs,
                                 graph_label=graph_label,
                                 draw_atomic_nodes=draw_atomic_nodes,
                                 draw_event_nodes=draw_event_nodes)
    dgraph = drawer.draw()
    dgraph.write(out_file_name, format='png')