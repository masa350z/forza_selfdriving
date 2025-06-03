# Forza Horizon 経路探索 Docker サーバー

このディレクトリは、**Forza Horizon 5 の地図とグラフ構造を用いた経路探索機能**を  
Docker コンテナとして常駐提供するための構成一式です。

経路探索処理を Python によって高速かつ軽量に常駐させ、  
座標入力 → edge ID 列出力 という形で利用できます。

---

## 📁 構成

```
docker-route/
├── Dockerfile              # Docker ビルド構成
├── requirements.txt        # 必要な Python パッケージ
├── data/                   # マップ・グラフデータ一式
│   ├── graphmap/
│   │   ├── graph.pickle
│   │   ├── movement_graph.pickle
│   │   ├── node.pickle
│   │   └── edge.pickle
│   └── nearest_edge_map_x1.npy
└── src/
    ├── config.py
    ├── coord2state.py
    └── route_cli.py        # メインスクリプト（標準入出力ルーティング）
```

---

## 🚀 ビルド

```bash
docker build -t forza-route ./docker-route
```

🔁 一時的に起動してルートを計算  
標準入力に JSON を与えて、標準出力で結果を受け取る形式：

```bash
echo '{"start_x":-500,"start_z":1200,"goal_x":3500,"goal_z":-2600}' | \
docker run -i --rm forza-route python src/route_cli.py
```

出力：

```json
{"edges": [136, 140, 287, ...]}
```

🛰️ コンテナを常駐させる（複数回呼びたい場合）

```bash
docker run -dit --name forza-route forza-route
```

その後、別プロセスから繰り返し呼び出す：

```bash
echo '{"start_x":-500,"start_z":1200,"goal_x":3500,"goal_z":-2600}' | \
docker exec -i forza-route python src/route_cli.py
```

またはファイルから：

```bash
echo '{"start_x":-500,"start_z":1200,"goal_x":3500,"goal_z":-2600}' > req.json
docker exec -i forza-route python src/route_cli.py < req.json
```

⏹️ 停止・削除

```bash
docker stop forza-route
docker rm forza-route
```

🔁 出力形式  
常に 1 行 JSON 形式（JSON Lines）で出力されます。

成功時：

```json
{"edges": [261, 298, 302, ...]}
```

エラー時：

```json
{"error": "座標がマップ範囲外です"}
```

📌 仕様補足  
コンテナ起動時に movement_graph.pickle などを一度だけロードして常駐します。

そのため 2回目以降のルート計算は非常に高速（1ミリ秒程度）です。

使用するマップデータはあらかじめ `make_graph_map.py` 等で生成済みのものを配置してください。

✅ 動作要件

- Docker Engine（Windows, WSL, Linux）  
- Forza Horizon のマップデータ生成済み  
- グラフ生成済み (`graphmap/`, `nearest_edge_map_x1.npy`)
