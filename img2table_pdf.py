from img2table.document import PDF

pdf = PDF(src="7d6ec5f6aca54d4b93eccd191c792bbb.pdf",
          pages=[22],
          detect_rotation=False,
          pdf_text_extraction=False)

# Extract tables
extracted_tables = pdf.extract_tables(
                                      implicit_rows=False,
                                      borderless_tables=False)

all_pdf_cells = []

for page, tables in extracted_tables.items():
    for table in tables:
        for row in table.content.values():
            for cell in row:
                x0, top, x1, bottom = cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2
                all_pdf_cells.append([x0, top, x1, bottom])

print(all_pdf_cells)          