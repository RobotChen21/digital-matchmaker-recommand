  # 基于 MCP 的动态地理空间匹配 (Dynamic Geospatial Matching via MCP)
  
  ## 1. 现状：基于静态知识库的模糊匹配 (The MVP)
  
  在目前的 MVP 版本中，我利用了大模型的通识推理能力来做初步的区域展开。
  *   **输入**：“上海周边”
  *   **LLM 推理**：基于训练数据的概率，理解为“苏州、杭州、无锡”等。
  *   **局限**：对于不需要精确距离的场景是够用的。但缺乏**实时地理感知**（无法处理“离我5km”），且难以处理**行政区划边界**（如“在上海上班，但别给我推昆山的人，虽然昆山离得近，但属于江苏，社保不通”）。
  
  ---
  
  ## 2. 进阶方案：基于 MCP 的 Agentic Geo-Matching (The Future)
  
  为了解决上述痛点，我规划了引入 **Model Context Protocol (MCP)** 理念，将地理位置服务（LBS）封装为标准的工具上下文，让 Agent 具备“时空推理”能力。
  
  ### 2.1 核心架构
  *   **Context Source**: 用户的实时 GPS 坐标（由前端/客户端透传）。
  *   **MCP Server**: 将高德/百度地图 API 封装为符合 MCP 标准的工具服务。
  *   **Tool Definition**: 暴露 `search_nearby_cities(lat, lon, radius)` 接口。
  
  ### 2.2 ReAct 动态推理工作流 (The Workflow)
  这是一个典型的 **Sense-Reason-Act** 循环，Agent 像人一样查地图、看边界、做决策。
  
  1.  **感知 (Sense)**
      *   **输入**: 用户 GPS (上海嘉定) + 语义需求 ("找附近的，但我是上海户口，别给我推江苏的，免得异地恋麻烦")。
      *   **意图分析**: LLM 识别出两个约束：① 物理距离近 (Radius)；② 行政区划排除 (Exclude Province=江苏)。
  
  2.  **行动 I (Act)**
      *   Agent 自主决定初始搜索半径（例如 20km），调用 MCP 工具 `search_nearby_cities(lat, lon, radius=20)`。
  
  3.  **推理与过滤 (Reason & Filter)**
      *   **观察 (Observation)**: 工具返回列表 `["嘉定区", "昆山市", "太仓市"]`。
      *   **思考 (Thought)**: "昆山和太仓虽然在 20km 内，但它们行政上属于江苏省。用户明确排除了江苏。必须剔除。"
      *   **过滤**: 执行逻辑删除，剩余 `["嘉定区"]`。
  
  4.  **自适应调整 (Self-Correction)**
      *   **评估**: 发现过滤后只剩 1 个区域，候选池太小。
      *   **再行动**: Agent 自动决策扩大半径至 50km，重新调用工具。
      *   **循环**: 直到获取满足约束的足够数量的候选城市。
  
  ---
  
  ## 3. 底层存储支撑：GeoSpatial Indexing
  
  为了支撑上述 Agent 决策后的高效查询，数据库层必须配套改造。我们不能只存字符串，必须存数学坐标。
  
  ### 3.1 数据层改造 (GeoJSON)
  在 MongoDB 的用户文档中，不再只存 `'city': '上海'`，而是存储标准的 GeoJSON 格式：
  
  ```json
  "location": {
    "type": "Point",
    "coordinates": [121.47, 31.23] // [经度, 纬度]
  }
  ```
  
  ### 3.2 索引优化 (`2dsphere`)
  在 MongoDB 中为 `location` 字段建立 `2dsphere` 索引。这是处理球体几何计算（地球表面距离）的标准索引，也是滴滴、美团等 LBS 应用的基石。
  
  ### 3.3 最终查询生成
  当 Agent 通过 MCP 流程确定了最终的搜索中心和半径（如：中心=[121.47, 31.23], 半径=50km）后，系统将其转化为 MongoDB 的 `$near` 查询：
  
  ```javascript
  db.users.find({
     location: {
        $near: {
           $geometry: { type: "Point", coordinates: [121.47, 31.23] },
           $maxDistance: 50000 // 单位米
        }
     }
  })
  ```
  