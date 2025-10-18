def create_pdf(kpis, plots, insights):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Data Dashboard Report", ln=True, align="C")
    pdf.ln(5)
    
    # KPIs
    pdf.set_font("Arial", size=12)
    for kpi in kpis:
        # Wrap text if too long
        try:
            pdf.multi_cell(0, 6, kpi)
        except Exception:
            # fallback: split manually
            for part in [kpi[i:i+80] for i in range(0, len(kpi), 80)]:
                pdf.multi_cell(0, 6, part)
    pdf.ln(5)
    
    # Insights
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    for line in insights:
        try:
            pdf.multi_cell(0, 6, line)
        except Exception:
            for part in [line[i:i+80] for i in range(0, len(line), 80)]:
                pdf.multi_cell(0, 6, part)
    pdf.ln(5)
    
    # Plots
    for name, fig in plots.items():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            fig.write_image(tmp_file.name, engine="kaleido")
            pdf.image(tmp_file.name, w=170)
        except Exception:
            continue
        finally:
            tmp_file.close()
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
    
    return pdf.output(dest="S").encode("latin-1", errors="ignore")
