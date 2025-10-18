File "/mount/src/smart-dashboard-generator/app.py", line 164, in <module>
    pdf_bytes = create_pdf(kpis_list, plots, ai_insights)
File "/mount/src/smart-dashboard-generator/app.py", line 134, in create_pdf
    pdf.multi_cell(0, 6, kpi)
    ~~~~~~~~~~~~~~^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/fpdf/fpdf.py", line 221, in wrapper
    return fn(self, *args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/fpdf/deprecation.py", line 32, in wrapper
    return fn(self, *args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/fpdf/fpdf.py", line 4208, in multi_cell
    text_line = multi_line_break.get_line()
File "/home/adminuser/venv/lib/python3.13/site-packages/fpdf/line_break.py", line 794, in get_line
    raise FPDFException(
        "Not enough horizontal space to render a single character"
    )
