from dgl.nn.pytorch import GraphConv
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import torch as th

class GraphConvEdgeWeight(GraphConv): # 기본dgl라이브러리에서 개선시킨 GCN 연산

    def forward(self, graph, feat,  weight=None, edge_weights=None):
        with graph.local_scope(): # 원본 그래프 구조에 영향을 주지 않음
            if not self._allow_zero_in_degree: # 0-in-degree node가 입력된 그래프에 있으면 에러 발생(노드에 메세지가 전달되지 않기 때문 -> mailbox에 아무것도 존재하지 않게 됨)
                if (graph.in_degrees() == 0).any(): # 0-in-degree node가 존재
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            # feat을 feat_src(소스 노드 feature)와 feat_dst(도착 노드 feature)로 확장하는 함수
            # 미니 배치 학습을 위해 샘플된 서브 그래프에 대해 
            # block: 서브그래프
            feat_src, feat_dst = expand_as_pair(feat, graph) 
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst