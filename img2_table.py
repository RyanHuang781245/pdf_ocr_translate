from img2table.document import Image
import cv2
from PIL import Image as PILImage

input_path = "output/pdf2img_outputs5/6cdca12396254927adb7937a06090039_p0023.png"

img = Image(src=input_path)

# Extract tables
extracted_tables = img.extract_tables(implicit_rows=False, implicit_columns=False, borderless_tables=False)

# Display extracted tables
table_img = cv2.imread(input_path)

all_pdf_cells = []

for table in extracted_tables:
    for row in table.content.values():
        for cell in row:
            x0, top, x1, bottom = cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2
            all_pdf_cells.append([x0, top, x1, bottom])

print(all_pdf_cells)