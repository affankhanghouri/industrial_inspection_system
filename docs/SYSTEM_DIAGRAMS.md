# Industrial Safety System Diagrams

This document is the visual map of the complete project. The diagrams use
[Mermaid](https://mermaid.js.org/) and render directly in GitHub, GitLab, and
Mermaid-enabled Markdown viewers.

## 1. System context

This view shows who uses the system and which external inputs it depends on.

```mermaid
flowchart LR
    Cameras["Three CCTV / MP4 camera sources"]
    HSE["HSE Officer"]
    Operator["System Operator / Developer"]

    System["Industrial AI Safety System"]
    Dashboard["Web Dashboard"]
    Preview["OpenCV Live Preview"]
    Evidence["Local Event Evidence"]

    Cameras -->|Video frames| System
    System --> Dashboard
    System --> Preview
    System --> Evidence
    HSE -->|Reviews, resolves, dismisses| Dashboard
    Operator -->|Starts and monitors| System
    Operator -->|Optional --display| Preview
```

## 2. High-level architecture

The live path is latest-only: camera and AI stages exchange the newest result
instead of building an unbounded processing queue.

```mermaid
flowchart LR
    subgraph Input["Video ingestion"]
        C1["CAM-001\nRefinery Area"]
        C2["CAM-002\nLoading Area"]
        C3["CAM-003\nRestricted Area"]
        Workers["CameraManager + CameraWorker threads"]
        LatestFrames["Per-camera LatestFrameBuffer\n1 frame each"]
        C1 & C2 & C3 --> Workers --> LatestFrames
    end

    subgraph Vision["Shared vision pipeline"]
        Person["Round-robin person inference\nONNX Runtime"]
        Tracking["Camera-local person tracking"]
        PPE["CAM-001 helmet inference\nOpenVINO + association"]
        Zones["CAM-002 / CAM-003\nzone evaluation"]
        LatestFrames --> Person --> Tracking
        Tracking --> PPE
        Tracking --> Zones
    end

    subgraph Safety["Safety decisions"]
        Coordinator["SafetyCoordinator"]
        Rules["RuleEngine\ntemporal confirmation + cooldown"]
        Events["EventService\ndeduplication + persistence"]
        PPE --> Coordinator
        Zones --> Coordinator
        Coordinator --> Rules --> Events
    end

    subgraph Outputs["Operations outputs"]
        SQLite[("SQLite safety.db")]
        Recorder["EvidenceRecorder\nbounded pre/post buffer"]
        Files["snapshot.jpg + clip.mp4"]
        API["FastAPI"]
        Web["HSE Dashboard"]
        Display["Single / multi-camera\nOpenCV preview"]
        Events --> SQLite
        Events --> Recorder --> Files
        SQLite --> API --> Web
        LatestFrames --> Display
        Tracking & PPE & Zones --> Display
    end
```

## 3. Low-level component design

This maps the main Python modules and the direction of their dependencies.

```mermaid
flowchart TB
    Launcher["scripts/run_system.py\ncomposition root + lifecycle"]

    subgraph VideoPkg["industrial_safety.video"]
        CameraConfig
        CameraManager
        CameraWorker
        MP4VideoSource
        LatestFrameBuffer
        FramePacket
        CameraManager --> CameraWorker
        CameraWorker --> MP4VideoSource
        CameraWorker --> LatestFrameBuffer
        LatestFrameBuffer --> FramePacket
    end

    subgraph InferencePkg["industrial_safety.inference"]
        Scheduler["RoundRobinInferenceScheduler"]
        PersonRuntime["ExportedPersonDetector\nONNX / OpenVINO adapter"]
        LatestInferenceResult
        Scheduler --> PersonRuntime
        Scheduler --> LatestInferenceResult
    end

    subgraph TrackingPkg["industrial_safety.tracking"]
        TrackingCoordinator
        Tracker["TimeAwareIoUPersonTracker"]
        LatestTrackingResult
        TrackingCoordinator --> Tracker
        TrackingCoordinator --> LatestTrackingResult
    end

    subgraph PPEPkg["industrial_safety.ppe"]
        PPECoordinator
        PPERuntime["ExportedPPEDetector\nOpenVINO adapter"]
        Associator["HelmetTrackAssociator"]
        LatestPPEAssociationResult
        PPECoordinator --> PPERuntime
        PPECoordinator --> Associator
        PPECoordinator --> LatestPPEAssociationResult
    end

    subgraph SafetyPkg["zones + rules + runtime"]
        ZoneEngine
        RuleEngine
        SafetyCoordinator
        SafetyCoordinator --> ZoneEngine
        SafetyCoordinator --> RuleEngine
    end

    subgraph EventPkg["events + storage"]
        EventService
        EvidenceRecorder
        Ring["EvidenceRingBuffer"]
        Store["SQLiteEventStore"]
        EventService --> EvidenceRecorder
        EvidenceRecorder --> Ring
        EventService --> Store
    end

    subgraph Presentation["api + presentation"]
        FastAPI["create_app"]
        Dashboard["web/templates + web/static"]
        Visualizer["draw_safety_preview"]
        Compositor["Multi-camera compositor"]
        FastAPI --> Dashboard
        Visualizer --> Compositor
    end

    Launcher --> CameraManager
    Launcher --> Scheduler
    Launcher --> TrackingCoordinator
    Launcher --> PPECoordinator
    Launcher --> SafetyCoordinator
    Launcher --> EventService
    Launcher --> FastAPI
    Launcher --> Visualizer

    LatestFrameBuffer --> Scheduler
    LatestInferenceResult --> TrackingCoordinator
    LatestTrackingResult --> PPECoordinator
    LatestTrackingResult --> SafetyCoordinator
    LatestPPEAssociationResult --> SafetyCoordinator
    RuleEngine --> EventService
    Store --> FastAPI
```

## 4. Runtime threads and bounded handoffs

Each producer can continue without waiting for slower consumers. Latest-result
buffers replace stale data, while only the evidence writer uses a queue—and that
queue is explicitly bounded.

```mermaid
flowchart LR
    subgraph Threads["Long-running threads"]
        CW1["camera-CAM-001"]
        CW2["camera-CAM-002"]
        CW3["camera-CAM-003"]
        Infer["person-inference"]
        Track["tracking-coordinator"]
        PPE["ppe-coordinator"]
        Safety["safety-event-coordinator"]
        Sampler["evidence-sampler"]
        Writer["evidence-writer"]
        API["dashboard-api"]
        Main["main / optional display"]
    end

    F1["Latest frame\n1 slot"]
    F2["Latest frame\n1 slot"]
    F3["Latest frame\n1 slot"]
    IR["Latest inference result"]
    TR["Latest tracking result"]
    PR["Latest PPE result"]
    RB["Evidence ring buffers\n25 frames/camera"]
    WQ["MP4 writer queue\nmax 8 jobs"]

    CW1 --> F1
    CW2 --> F2
    CW3 --> F3
    F1 & F2 & F3 --> Infer --> IR --> Track --> TR
    TR --> PPE --> PR
    TR & PR --> Safety
    F1 & F2 & F3 --> Sampler --> RB
    Safety -->|Confirmed event trigger| RB
    RB --> WQ --> Writer
    Safety --> API
    F1 & F2 & F3 --> Main
    TR & PR --> Main
```

## 5. Camera-specific safety paths

```mermaid
flowchart TB
    Frames["Person tracks"]

    subgraph HelmetPath["CAM-001: helmet required"]
        H1["PPE detector samples frame"]
        H2["Associate helmet boxes to track head regions"]
        H3{"Helmet associated?"}
        H4["OBSERVED\nclear pending absence"]
        H5["UNOBSERVED\nadd temporal observation"]
        H6{"At least 2 unobserved\nwithin 3.5 seconds?"}
        H7["HELMET_MISSING"]
        H1 --> H2 --> H3
        H3 -->|Yes| H4
        H3 -->|No| H5 --> H6 -->|Yes| H7
    end

    subgraph LoadingPath["CAM-002: loading area"]
        L1["Track bottom-center point"]
        L2["Evaluate loading polygon"]
        L3["Confirm outside-to-inside transition"]
        L4["LOADING_AREA_ENTRY"]
        L1 --> L2 --> L3 --> L4
    end

    subgraph RestrictedPath["CAM-003: restricted doorway"]
        R1["Track bottom-center point"]
        R2["Evaluate restricted polygon"]
        R3["Confirm outside-to-inside transition"]
        R4["RESTRICTED_AREA_ENTRY"]
        R1 --> R2 --> R3 --> R4
    end

    Frames --> H1
    Frames --> L1
    Frames --> R1
    H7 & L4 & R4 --> Cooldown["Per-rule, per-camera\n300-second cooldown"]
```

## 6. Confirmed-event and evidence sequence

```mermaid
sequenceDiagram
    participant Camera as CameraWorker
    participant Buffer as EvidenceRingBuffer
    participant Vision as Tracking / PPE / Zone
    participant Rules as RuleEngine
    participant Events as EventService
    participant DB as SQLiteEventStore
    participant Writer as Evidence writer

    loop At 10 evidence FPS
        Camera->>Buffer: Retain newest sampled frame
        Note over Buffer: Maximum 25 frames per camera
    end

    Camera->>Vision: Publish latest frame
    Vision->>Rules: PPE observation or zone transition
    Rules->>Rules: Apply temporal confirmation
    Rules->>Events: Confirmed RuleMatch
    Events->>DB: Check latest event for cooldown

    alt Duplicate inside cooldown
        Events-->>Rules: Suppress duplicate
    else New confirmed event
        Events->>Events: Generate event ID and annotate snapshot
        Events->>Writer: Trigger clip with pre-event buffer
        Events->>DB: Insert OPEN event with snapshot and clip paths
        loop For up to 2.5 seconds after event
            Camera->>Buffer: Add post-event samples
        end
        Buffer->>Writer: Bounded background encoding job
        Writer->>Writer: Write one MP4 and release temporary frames
    end
```

## 7. Event lifecycle

Detection state and human review state are separate. A weak observation never
becomes a database event until the configured rule confirms it.

```mermaid
stateDiagram-v2
    [*] --> NormalMonitoring
    NormalMonitoring --> PendingConfirmation: Rule receives qualifying observation
    PendingConfirmation --> NormalMonitoring: Observation clears / confirmation expires
    PendingConfirmation --> ConfirmedEvent: Temporal or zone confirmation succeeds
    ConfirmedEvent --> Cooldown: Event persisted once
    Cooldown --> NormalMonitoring: Cooldown expires

    state ConfirmedEvent {
        [*] --> OPEN
        OPEN --> RESOLVED: HSE officer resolves
        OPEN --> DISMISSED: HSE officer dismisses
        RESOLVED --> DISMISSED: Optional later dismissal
    }
```

## 8. Database and evidence model

```mermaid
erDiagram
    SAFETY_EVENT {
        string event_id PK
        string event_type
        string status
        string severity
        datetime detected_at
        string camera_id
        string zone_id "nullable"
        int track_id
        string rule_id
        string rule_version
        string person_model_version
        string ppe_model_version "nullable"
        float confidence "nullable"
        int supporting_observations
        string snapshot_path
        string clip_path "nullable"
        datetime created_at
        datetime reviewed_at "nullable"
        string review_action "nullable"
        string review_reason "nullable"
    }

    EVIDENCE_DIRECTORY {
        string date
        string event_id
        file snapshot_jpg
        file clip_mp4
    }

    SAFETY_EVENT ||--|| EVIDENCE_DIRECTORY : "references paths in"
```

Physical evidence layout:

```text
data/evidence/
└── YYYY-MM-DD/
    └── <event-id>/
        ├── snapshot.jpg
        └── clip.mp4
```

## 9. API and dashboard flow

```mermaid
flowchart LR
    Browser["HSE browser"]
    Static["HTML + CSS + JavaScript"]
    API["FastAPI routes"]
    Health["Live camera health provider"]
    Store[("SQLiteEventStore")]
    Evidence["Path-confined evidence files"]

    Browser -->|GET /| Static
    Static -->|GET /api/cameras| API --> Health
    Static -->|GET /api/stats| API --> Store
    Static -->|GET /api/events| API --> Store
    Static -->|GET event screenshot| API --> Evidence
    Static -->|GET event clip / byte range| API --> Evidence
    Static -->|POST resolve| API --> Store
    Static -->|POST dismiss or ×| API --> Store

    Store -->|Event JSON| API --> Static
    Evidence -->|JPEG / MP4| API --> Static
```

## 10. Display-mode decision flow

The OpenCV display is a read-only presentation consumer. It does not feed data
back into detection, tracking, or safety decisions.

```mermaid
flowchart TD
    Start["run_system.py"] --> Display{"--display supplied?"}
    Display -->|No| Headless["Run three-camera pipeline headlessly"]
    Display -->|Yes| Camera{"--camera supplied?"}
    Camera -->|Yes| Single["Start selected camera only\nSingle live window"]
    Camera -->|No| Multi["Start all active cameras\nOne multi-camera window"]

    Multi --> Independent["Read each latest-frame buffer independently"]
    Independent --> Render["Reuse draw_safety_preview per camera"]
    Render --> Tile["CAM-001 and CAM-002 top\nCAM-003 centered below"]
    Tile --> Controls["q / window close / Ctrl+C"]
    Single --> Controls
    Controls --> Shutdown["Ordered graceful component shutdown"]
```

## 11. Startup and shutdown order

```mermaid
flowchart LR
    subgraph Startup
        A["Load YAML configuration"] --> B["Warm person detector"]
        B --> C["Warm PPE detector if enabled"]
        C --> D["Create stores, engines, coordinators"]
        D --> E["Start camera workers"]
        E --> F["Start evidence sampler"]
        F --> G["Start inference"]
        G --> H["Start tracking"]
        H --> I["Start PPE"]
        I --> J["Start safety coordinator"]
        J --> K["Start API and optional display"]
    end

    subgraph Shutdown
        S1["Stop safety coordinator"] --> S2["Flush evidence recorder"]
        S2 --> S3["Stop PPE"]
        S3 --> S4["Stop tracking"]
        S4 --> S5["Stop inference"]
        S5 --> S6["Stop camera workers"]
        S6 --> S7["Stop API / close display"]
        S7 --> S8["Close SQLite"]
    end
```

## Reading order

For a quick understanding, read diagrams **1 → 2 → 5 → 6 → 9**. Developers
changing implementation details should additionally use diagrams **3, 4, 8,
10, and 11**.
