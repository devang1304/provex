# PROVEX Makefile

.PHONY: explain node_mapping report dashboard clean

# Run explanations pipeline
explain:
	python -m explanations.window_analysis

# Export node ID → label mapping from PostgreSQL
node_mapping:
	python -m explanations.export_node_mapping

# Generate report
report:
	python -m reporting.generate_report

# Launch dashboard
dashboard:
	streamlit run reporting/streamlit_dashboard.py

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
