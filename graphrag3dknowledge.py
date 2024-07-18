import os #用于文件系统操作
import pandas as pd #用于数据处理和操作
import networkx as nx #用于创建和分析图结构
import plotly.graph_objects as go #plotly：用于创建交互式可视化 plotly.graph_objects：用于创建低级的plotly图形对象
from plotly.subplots import make_subplots #用于创建子图
import plotly.express as px #用于快速创建统计图表

def read_parquet_files(directory):
    """
    读取指定目录下的所有Parquet文件并合并
    功能：读取指定目录下的所有Parquet文件并合并成一个DataFrame
    实现：使用os.listdir遍历目录，pd.read_parquet读取每个文件，然后用pd.concat合并
    """
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            file_path = os.path.join(directory, filename)
            df = pd.read_parquet(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def clean_dataframe(df):
    """
    清理DataFrame，移除无效的行
    功能：清理DataFrame，移除无效的行
    实现：删除source和target列中的空值，将这两列转换为字符串类型
    """
    df = df.dropna(subset=['source', 'target'])
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
    return df


def create_knowledge_graph(df):
    """
    从DataFrame创建知识图谱
    功能：从DataFrame创建知识图谱
    实现：使用networkx创建有向图，遍历DataFrame的每一行，添加边和属性
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        attributes = {k: v for k, v in row.items() if k not in ['source', 'target']}
        G.add_edge(source, target, **attributes)
    return G


def create_node_link_trace(G, pos):
    """
    功能：创建节点和边的3D轨迹
    实现：使用networkx的布局信息创建Plotly的Scatter3d对象
    """
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'Node: {node}<br># of connections: {len(adjacencies)}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    return edge_trace, node_trace


def create_edge_label_trace(G, pos, edge_labels):
    """
    功能：创建边标签的3D轨迹
    实现：计算边的中点位置，创建Scatter3d对象显示标签
    """
    return go.Scatter3d(
        x=[pos[edge[0]][0] + (pos[edge[1]][0] - pos[edge[0]][0]) / 2 for edge in edge_labels],
        y=[pos[edge[0]][1] + (pos[edge[1]][1] - pos[edge[0]][1]) / 2 for edge in edge_labels],
        z=[pos[edge[0]][2] + (pos[edge[1]][2] - pos[edge[0]][2]) / 2 for edge in edge_labels],
        mode='text',
        text=list(edge_labels.values()),
        textposition='middle center',
        hoverinfo='none'
    )


def create_degree_distribution(G):
    """
    功能：创建节点度分布直方图
    实现：使用plotly.express创建直方图
    """
    degrees = [d for n, d in G.degree()]
    fig = px.histogram(x=degrees, nbins=20, labels={'x': 'Degree', 'y': 'Count'})
    fig.update_layout(
        title_text='Node Degree Distribution',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    return fig


def create_centrality_plot(G):
    """
    功能：创建节点中心性分布箱线图
    实现：计算度中心性，使用plotly.express创建箱线图
    """
    centrality = nx.degree_centrality(G)
    centrality_values = list(centrality.values())
    fig = px.box(y=centrality_values, labels={'y': 'Centrality'})
    fig.update_layout(
        title_text='Degree Centrality Distribution',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    return fig


def visualize_graph_plotly(G):
    """功能：使用Plotly创建全面优化布局的高级交互式知识图谱可视化
    实现：
        创建3D布局
        生成节点和边的轨迹
        创建子图，包括3D图、度分布图和中心性分布图
        添加交互式按钮和滑块
        优化整体布局
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty. Nothing to visualize.")
        return

    pos = nx.spring_layout(G, dim=3)  # 3D layout
    edge_trace, node_trace = create_node_link_trace(G, pos)

    edge_labels = nx.get_edge_attributes(G, 'relation')
    edge_label_trace = create_edge_label_trace(G, pos, edge_labels)

    degree_dist_fig = create_degree_distribution(G)
    centrality_fig = create_centrality_plot(G)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.7, 0.3],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}]
        ],
        subplot_titles=("3D Knowledge Graph Code by AI超元域频道", "Node Degree Distribution", "Degree Centrality Distribution")
    )

    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(edge_label_trace, row=1, col=1)

    fig.add_trace(degree_dist_fig.data[0], row=1, col=2)
    fig.add_trace(centrality_fig.data[0], row=2, col=2)

    # Update 3D layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            aspectmode='cube'
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )

    # Add buttons for different layouts
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(args=[{"visible": [True, True, True, True, True]}], label="Show All", method="update"),
                    dict(args=[{"visible": [True, True, False, True, True]}], label="Hide Edge Labels",
                         method="update"),
                    dict(args=[{"visible": [False, True, False, True, True]}], label="Nodes Only", method="update")
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Add slider for node size
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Node Size: "},
            pad={"t": 50},
            steps=[dict(method='update',
                        args=[{'marker.size': [i] * len(G.nodes)}],
                        label=str(i)) for i in range(5, 21, 5)]
        )]
    )

    # 优化整体布局
    # fig.update_layout(
    #     height=1198,  # 增加整体高度
    #     width=2055,  # 增加整体宽度
    #     title_text="Advanced Interactive Knowledge Graph",
    #     margin=dict(l=10, r=10, t=25, b=10),
    #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    # )

    fig.show()


def main():
    """ 功能：主函数，协调整个程序的执行流程
        实现：
            读取Parquet文件
            清理数据
            创建知识图谱
            打印图的统计信息
            调用可视化函数
    """
    directory = '/Users/charlesqin/PycharmProjects/RAGCode/inputs/artifacts'  # 替换为实际的目录路径
    df = read_parquet_files(directory)

    if df.empty:
        print("No data found in the specified directory.")
        return

    print("Original DataFrame shape:", df.shape)
    print("Original DataFrame columns:", df.columns.tolist())
    print("Original DataFrame head:")
    print(df.head())

    df = clean_dataframe(df)

    print("\nCleaned DataFrame shape:", df.shape)
    print("Cleaned DataFrame head:")
    print(df.head())

    if df.empty:
        print("No valid data remaining after cleaning.")
        return

    G = create_knowledge_graph(df)

    print(f"\nGraph statistics:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        print(f"Connected components: {nx.number_connected_components(G.to_undirected())}")
        visualize_graph_plotly(G)
    else:
        print("Graph is empty. Cannot visualize.")


if __name__ == "__main__":
    main()
