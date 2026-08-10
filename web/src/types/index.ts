export type DocStatus = 'queued' | 'processing' | 'done' | 'failed'

export interface DocumentRow {
    id: number
    filename: string
    size_bytes: number
    status: DocStatus
    doc_type: string | null
    doc_type_base: string | null
    doc_type_era: string | null
    recognised: boolean
    doc_conf: number | null
    /** Denormalised verdicts; vocabulary varies ('good'/'bad', 'REAL'/'FAKE'). */
    quality: Record<string, string | number>
    field_count: number
    device: string | null
    processing_ms: number | null
    error: string | null
    error_code: string | null
    retry_count: number
    has_canvas: boolean
    created_at: string
    started_at: string | null
    finished_at: string | null
    queue_position?: number | null
}

/** Axis-aligned box in CANVAS pixel space (see DocumentDetail.coord_space). */
export interface Box {
    id: string
    label: string
    display: string
    kind: 'text' | 'visual'
    x1: number
    y1: number
    x2: number
    y2: number
    conf: number | null
    cls: number | null
    text: string | null
    /** Another box with the same label owns the recognised text. */
    ambiguous: boolean
}

export interface OrientedBox {
    cx: number
    cy: number
    w: number
    h: number
    angle_rad: number
    conf: number | null
    label: string | null
}

export interface AddressLine {
    id: string
    kind: string | null
    text: string | null
    p_handwritten: number | null
    obbox: OrientedBox | null
}

export interface AddressBlock {
    /** False when the geometry and the text lists desynchronised; do not draw. */
    aligned: boolean
    lines: AddressLine[]
}

export interface Field {
    name: string
    display: string
    value: string | null
    script: 'ru' | 'en' | 'num'
    conf: number | null
    box_ids: string[]
}

export interface DocumentDetail extends DocumentRow {
    canvas: { url: string; width: number | null; height: number | null; is_fallback?: boolean }
    original: { url: string; width: number | null; height: number | null; content_type: string }
    coord_space: string | null
    coord_space_note: string | null
    boxes: Box[]
    fields: Field[]
    ocr: Record<string, string>
    quality: Record<string, string | number>
    timings: Record<string, number>
    address: AddressBlock | null
}

export interface DocumentListResponse {
    items: DocumentRow[]
    total: number
    page: number
    page_size: number
    stats: Record<string, number | null>
}

export interface Progress {
    step: string
    label: string
    pct: number
    eta_sec: number | null
    queue_position: number | null
}

export interface ApiKeyRow {
    id: number
    label: string
    prefix: string
    masked: string
    is_default: boolean
    /** True when the service invented this key because DEFAULT_API_KEY was unset. */
    is_generated?: boolean
    created_at: string
    last_used_at: string | null
    /** Present only in the create response — shown once, never retrievable. */
    key?: string
}

export interface SettingDef {
    key: string
    type: 'bool' | 'int' | 'float' | 'choice' | 'str'
    default: string
    label: string
    description: string
    group: string
    min_value: number | null
    max_value: number | null
    choices: string[] | null
    restart_required: boolean
}

export interface DocumentFilter {
    page: number
    page_size: number
    search: string
    status: string
    doc_type: string
    date_from: string
    date_to: string
    sort_by: string
    sort_dir: 'asc' | 'desc'
}
