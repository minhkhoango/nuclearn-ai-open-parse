import logging
import time
from typing import Any, Dict, List, Literal

import torch  # type: ignore
from PIL import Image  # type: ignore
from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection,
    TableTransformerForObjectDetection,
)

# type: ignore
from openparse.config import config

from ..schemas import (
    BBox,
    Size,
)
from ..utils import (
    convert_croppped_cords_to_full_img_cords,
    convert_img_cords_to_pdf_cords,
    crop_img_with_padding,
    display_cells_on_img,
)
from .geometry import (
    calc_bbox_intersection,
)
from .schemas import (
    Table,
    TableCellModelOutput,
    TableDataCell,
    TableHeader,
    TableHeaderCell,
    TableRow,
)

# ================================
# === GLOBAL SETUP & CONSTANTS ===
# ================================
t0: float = time.time()
device: Literal['cuda'] | Literal['cpu'] = config.get_device()


# ==========================
# === ML MODEL ABSTRACTION ===
# ==========================
class TableDetector:
    """
    A wrapper for loading and running table detection models.
    """
    
    model: AutoModelForObjectDetection
    processor: AutoImageProcessor
    device: torch.device | str
    
    def __init__(self, model_id: str, device: torch.device | str) -> None:
        self.device = device
        print(f"Loading table detection model: {model_id}")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(self.device)

    def detect(self, image: Image.Image, threshold: float) -> List[BBox]:
        """
        Runs table detection on a PIL image.
        Returns a list of bounding boxes for detected tables.
        """
        inputs: Dict[str, torch.Tensor] = self.processor(
            images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs: Any = self.model(**inputs)

        target_sizes: torch.Tensor = torch.tensor([image.size[::-1]]).to(self.device)
        results: Dict[str, torch.Tensor] = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        tables_found: List[BBox] = []
        scores: torch.Tensor = results["scores"]
        labels: torch.Tensor = results["labels"]
        boxes: torch.Tensor = results["boxes"]

        for _score, _label, box_tensor in zip(scores, labels, boxes):
            box: List[float] = [round(i, 2) for i in box_tensor.tolist()]
            tables_found.append(tuple(box))  # type: ignore

        return tables_found


# ==================================
# === MODEL LOADING (STRUCTURE) ===
# ==================================

# --- This model recognizes the internal structure (rows, cells) of a table ---
STRUCTURE_MODEL_ID: str = "microsoft/table-transformer-structure-recognition"
structure_processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(
    STRUCTURE_MODEL_ID, revision="no_timm"
)
structure_model: TableTransformerForObjectDetection = (
    TableTransformerForObjectDetection.from_pretrained(
        STRUCTURE_MODEL_ID, revision="no_timm"
    ).to(device)
)

logging.info(f"Models loaded successfully 🚀: {time.time() - t0:.2f}s")


##################################
### === ML TABLE DETECTION === ###
##################################

# Adapted from:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Inference_with_Table_Transformer_(TATR)_for_parsing_tables.ipynb


def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Converts a bounding box format from center coordinates (cx, cy, width, height) to
    boundary coordinates (x_min, y_min, x_max, y_max).

    Parameters:
    - x: A tensor of shape (N, 4) representing N bounding boxes in cx, cy, w, h format.

    Returns:
    - A tensor of shape (N, 4) representing N bounding boxes in x_min, y_min, x_max, y_max format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b: List[torch.Tensor] = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# ==================================
# === STRUCTURED CONTENT PARSING ===
# ==================================

def _calculate_area(bbox: BBox | None) -> float:
    if bbox is None:
        return 0
    width: float = bbox[2] - bbox[0]
    height: float = bbox[3] - bbox[1]
    return width * height


def _is_overlapping_with_headers(
    cell_bbox: BBox, headers: List[TableHeader], overlap_threshold: float = 0.9
) -> bool:
    """
    Check if a given cell's bounding box overlaps with any of the header cells' bounding boxes.
    If the overlap area is above the threshold percentage of the cell's area, return True.
    """
    cell_area: float = _calculate_area(cell_bbox)
    if cell_area == 0:
        return False

    for header in headers:
        for hcell in header.cells:
            intersection: BBox | None = calc_bbox_intersection(cell_bbox, hcell.bbox)
            if intersection:
                intersection_area: float = _calculate_area(intersection)
                if (intersection_area / cell_area) > overlap_threshold:
                    return True
    return False

def _preprocess_header_cells(
    header_rows: List[TableCellModelOutput],
    cols: List[TableCellModelOutput],
    image_size: Size,
    page_size: Size,
) -> List[TableHeader]:
    """Processes detected header cells."""
    header_objs: List[TableHeader] = []
    for header_row in header_rows:
        header_row_cells: List[TableHeaderCell] = []
        for col in cols:
            cell_bbox: BBox | None = calc_bbox_intersection(
                header_row.bbox, col.bbox, safety_margin=5
            )
            if cell_bbox:
                pdf_cell_bbox: BBox = convert_img_cords_to_pdf_cords(
                    cell_bbox, page_size, image_size
                )
                header_row_cells.append(TableHeaderCell(bbox=pdf_cell_bbox))
        if header_row_cells:
            header_objs.append(TableHeader(cells=header_row_cells))
    return header_objs


def _process_row_cells(
    rows: List[TableCellModelOutput],
    cols: List[TableCellModelOutput],
    headers: List[TableHeader],
    image_size: Size,
    page_size: Size,
) -> List[TableRow]:
    """Processes detected data rows, avoiding overlaps with headers."""
    data_rows: List[TableRow] = []
    for row in rows:
        row_cells: List[TableDataCell] = []
        for col in cols:
            cell_bbox: BBox | None = calc_bbox_intersection(
                row.bbox, col.bbox, safety_margin=5
            )
            if cell_bbox:
                pdf_cell_bbox: BBox = convert_img_cords_to_pdf_cords(
                    cell_bbox, page_size, image_size
                )
                if not _is_overlapping_with_headers(pdf_cell_bbox, headers):
                    row_cells.append(TableDataCell(bbox=pdf_cell_bbox))
        if row_cells:
            data_rows.append(TableRow(cells=row_cells))
    return data_rows


def _structure_outputs_to_cells(
    outputs: Any, img_size: Size, id2label: Dict[int, str]
) -> List[TableCellModelOutput]:
    """Converts raw structure model outputs to a list of TableCellModelOutput objects."""
    logits: torch.Tensor = outputs.logits
    bboxes: torch.Tensor = outputs.pred_boxes

    # Post-process
    m: torch.return_types.max = logits.softmax(-1).max(-1)
    pred_labels: List[int] = m.indices.detach().cpu().numpy().tolist()[0]
    pred_scores: List[float] = m.values.detach().cpu().numpy().tolist()[0]
    pred_bboxes_raw: torch.Tensor = bboxes.detach().cpu()[0]

    # Rescale bboxes
    width, height = img_size
    box_scaler: torch.Tensor = torch.tensor([width, height, width, height])
    pred_bboxes_scaled: List[List[float]] = (
        _box_cxcywh_to_xyxy(pred_bboxes_raw) * box_scaler
    ).tolist()

    cells: List[TableCellModelOutput] = []
    for label_id, score, bbox in zip(pred_labels, pred_scores, pred_bboxes_scaled):
        class_label: str = id2label.get(label_id, "no object")
        if class_label != "no object":
            cells.append(
                TableCellModelOutput(
                    label=class_label,  # type: ignore
                    confidence=score,
                    bbox=tuple(bbox),  # type: ignore
                )
            )
    return cells


def table_from_model_outputs(
    image: Image.Image,
    page_size: Size,
    table_bbox: BBox,
    table_cells: List[TableCellModelOutput],
    min_cell_confidence: float,
) -> "Table":
    """Constructs a _Table object from the processed model outputs."""
    headers_raw: List[TableCellModelOutput] = [
        cell for cell in table_cells if cell.is_header and cell.confidence > min_cell_confidence
    ]
    rows_raw: List[TableCellModelOutput] = [
        cell for cell in table_cells if cell.is_row and cell.confidence > min_cell_confidence
    ]
    cols_raw: List[TableCellModelOutput] = [
        cell for cell in table_cells if cell.is_column and cell.confidence > min_cell_confidence
    ]

    header_objs: List[TableHeader] = _preprocess_header_cells(
        headers_raw, cols_raw, image.size, page_size
    )
    row_objs: List[TableRow] = _process_row_cells(
        rows_raw, cols_raw, header_objs, image.size, page_size
    )

    return Table(bbox=table_bbox, headers=header_objs, rows=row_objs)

def get_table_content(
    page_dims: Size,
    page_img: Image.Image,
    table_bbox: BBox,
    min_cell_confidence: float,
    verbose: bool = False,
) -> Table:
    """
    Crops a table from a page image, recognizes its structure, and returns a _Table object.
    """
    OFFSET: float = 0.05
    table_img: Image.Image = crop_img_with_padding(
        page_img, table_bbox, padding_pct=OFFSET
    )
    structure_id2label: Dict[int, str] = {
        **structure_model.config.id2label,
        len(structure_model.config.id2label): "no object",
    }

    # Use the dedicated processor for the structure model
    pixel_values_st: torch.Tensor = structure_processor(
        table_img, return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        outputs_st: Any = structure_model(pixel_values_st)

    # We need to re-implement the post-processing since we can't use the AutoProcessor's method here
    # This requires the old helper functions, so we will define them locally or bring them back
    # For now, let's assume a function that does this.
    cells: List[TableCellModelOutput] = _structure_outputs_to_cells(
        outputs_st, table_img.size, structure_id2label
    )

    for cell in cells:
        cell.bbox = convert_croppped_cords_to_full_img_cords(
            padding_pct=OFFSET,
            cropped_image_size=table_img.size,
            table_bbox=table_bbox,
            bbox=cell.bbox,
        )

    if verbose:
        display_cells_on_img(
            page_img, cells, "all", min_cell_confidence=min_cell_confidence
        )

    return table_from_model_outputs(
        page_img, page_dims, table_bbox, cells, min_cell_confidence
    )
