# Frontend Guidelines

This document outlines the guidelines and best practices for developing the frontend components of the Knowledge Graph Synthesis System. Since we're using a Streamlit-based approach, these guidelines focus on creating a simple, intuitive, and functional interface.

## 1. Frontend Philosophy

### 1.1 Simplicity First
- Focus on functionality over aesthetics
- Use built-in Streamlit components when possible
- Avoid unnecessary complexity or customization

### 1.2 Progressive Disclosure
- Start with simple options, hide advanced features until needed
- Use expanders for detailed information
- Layer information from high-level to detailed

### 1.3 User-Centric Design
- Provide clear feedback for long-running operations
- Show progress indicators for multi-step processes
- Prevent user errors through validation and guidance

## 2. Streamlit Implementation

### 2.1 Project Structure
```
src/
  frontend/
    app.py                # Main application entry point
    pages/                # Multi-page app components
      home.py
      config.py
      analysis.py
      results.py
    components/           # Reusable UI components
      graph_viewer.py
      text_processor.py
      result_formatter.py
    utils/                # Frontend utilities
      st_helpers.py       # Streamlit helper functions
      graph_visualization.py
```

### 2.2 State Management
- Use Streamlit's session state for persistent data between reruns
- Store large objects (like graphs) in session state
- Clear appropriate state variables when starting new processes

```python
# Example state management
if 'graph' not in st.session_state:
    st.session_state.graph = None

if st.button('Process New Text'):
    # Clear relevant state
    st.session_state.graph = None
    st.session_state.results = None
    # Process new text...
```

### 2.3 Layout Guidelines
- Use columns for side-by-side elements
- Use containers for logical grouping
- Use tabs for alternative views of the same data
- Ensure adequate spacing between elements

```python
# Example layout
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    # Input controls...

with col2:
    st.header("Preview")
    # Preview content...

with st.container():
    st.subheader("Results")
    tab1, tab2, tab3 = st.tabs(["Text", "Graph", "Theories"])
    # Tab content...
```

## 3. UI Components

### 3.1 Text Input
- Support both direct text entry and file upload
- Show character count and limits
- Provide sample texts for quick testing

```python
text_source = st.radio("Text Source", ["Enter Text", "Upload File", "Sample Text"])

if text_source == "Enter Text":
    text = st.text_area("Enter your text", height=300)
elif text_source == "Upload File":
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    # Process file...
else:
    sample = st.selectbox("Select sample", ["Scientific Paper", "Literary Work", "Interview"])
    # Load sample...
```

### 3.2 Configuration Options
- Group related settings
- Use appropriate controls for each setting type
- Provide sensible defaults and explanations

```python
with st.expander("LLM Settings"):
    provider = st.selectbox("LLM Provider", ["Claude", "GPT", "Gemini", "DeepSeek", "Ollama"])
    
    if provider == "Ollama":
        model = st.selectbox("Model", ["llama2", "mistral", "vicuna"])
    elif provider == "Claude":
        model = st.selectbox("Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229"])
    # Other provider options...
```

### 3.3 Process Control
- Clear buttons for starting processes
- Ability to cancel long-running operations
- Progress indicators for multi-step operations

```python
if st.button("Start Analysis"):
    progress = st.progress(0)
    status = st.empty()
    
    for i, step in enumerate(analysis_steps):
        status.text(f"Processing: {step}")
        # Perform step...
        progress.progress((i + 1) / len(analysis_steps))
```

### 3.4 Results Display
- Use tabs for different view types
- Provide options to download results
- Enable interactive exploration of graphs

```python
with st.expander("Theory 1: Pattern Recognition", expanded=True):
    st.markdown(theory_description)
    
    with st.container():
        st.subheader("Supporting Evidence")
        for evidence in supporting_evidence:
            st.markdown(f"- {evidence}")
    
    with st.container():
        st.subheader("Visual Representation")
        # Display relevant visualization
```

### 3.5 Graph Visualization
- Use PyVis for interactive graph visualization
- Provide filtering options to focus on specific parts
- Include zoom, pan, and search functionality

```python
def visualize_graph(graph):
    # Generate visualization HTML
    vis_html = create_pyvis_html(graph)
    
    # Display in Streamlit
    components.html(vis_html, height=600)
    
    # Graph controls
    with st.expander("Visualization Controls"):
        min_connections = st.slider("Min Connections", 1, 10, 1)
        highlight_nodes = st.multiselect("Highlight Nodes", list(graph.nodes))
        # Apply filters and update visualization...
```

## 4. Internationalization

### 4.1 Text Externalization
- Store UI strings in separate files
- Support both Russian and English interfaces
- Use a simple translation function

```python
# translations.py
TEXTS = {
    "en": {
        "start_button": "Start Analysis",
        "config_title": "Configuration",
        # ...
    },
    "ru": {
        "start_button": "Начать анализ",
        "config_title": "Конфигурация",
        # ...
    }
}

# Usage
lang = st.session_state.get("language", "en")
texts = TEXTS[lang]
st.button(texts["start_button"])
```

### 4.2 Language Switcher
- Provide a simple language toggle
- Persist language choice in session state
- Update all UI elements when language changes

```python
lang_options = {"English": "en", "Русский": "ru"}
selected_lang = st.sidebar.selectbox("Language/Язык", list(lang_options.keys()))

if "language" not in st.session_state or lang_options[selected_lang] != st.session_state.language:
    st.session_state.language = lang_options[selected_lang]
    st.experimental_rerun()
```

## 5. Error Handling

### 5.1 User Input Validation
- Validate inputs before processing
- Provide clear error messages
- Suggest corrections when possible

```python
if not text.strip():
    st.error("Please enter some text to analyze.")
    st.stop()

if len(text) < 100:
    st.warning("Text might be too short for meaningful analysis. Consider providing more content.")
```

### 5.2 Process Errors
- Catch and display errors from backend processes
- Provide context and suggested actions
- Log detailed errors for debugging

```python
try:
    result = process_text(text)
except APIError as e:
    st.error(f"API Error: {str(e)}")
    st.info("Please check your API key and try again.")
    log_error(e)  # Log detailed error
    st.stop()
except Exception as e:
    st.error("An unexpected error occurred.")
    with st.expander("Error Details"):
        st.write(str(e))
    log_error(e)  # Log detailed error
    st.stop()
```

### 5.3 Empty States
- Show helpful information when no data is available
- Provide clear next steps
- Use placeholders for future content

```python
if not st.session_state.get("graph"):
    st.info("No graph available. Process a text to generate a knowledge graph.")
    
    with st.expander("Sample Texts"):
        st.write("Try one of these sample texts to get started:")
        if st.button("Scientific Abstract"):
            st.session_state.text = SAMPLE_TEXTS["scientific"]
            st.experimental_rerun()
        # Other samples...
```

## 6. Performance Considerations

### 6.1 Caching
- Use Streamlit's caching for expensive operations
- Cache intermediate results to improve responsiveness
- Clear cache when inputs change

```python
@st.cache_data
def process_large_text(text, params):
    # Expensive processing...
    return result

# Usage
result = process_large_text(text, params)
```

### 6.2 Lazy Loading
- Load data only when needed
- Use expanders to defer content rendering
- Split large visualizations into separate tabs

```python
with st.expander("Detailed Analysis", expanded=False):
    # This content is only processed when expanded
    st.write("Detailed metrics:")
    detailed_analysis = compute_detailed_metrics(graph)  # Only runs when expanded
    st.dataframe(detailed_analysis)
```

### 6.3 Progress Feedback
- Show progress for long-running operations
- Provide estimates of remaining time when possible
- Allow background processing with status updates

```python
def process_with_progress():
    progress = st.progress(0)
    status_text = st.empty()
    
    for i, step in enumerate(steps):
        status_text.text(f"Step {i+1}/{len(steps)}: {step}")
        # Process step...
        progress.progress((i + 1) / len(steps))
    
    status_text.text("Processing complete!")
    return result
```

## 7. Accessibility

### 7.1 Color Considerations
- Use color combinations with sufficient contrast
- Don't rely solely on color to convey information
- Support Streamlit's light and dark themes

### 7.2 Text Readability
- Use adequate font sizes
- Maintain proper heading hierarchy
- Use clear, concise language

### 7.3 Keyboard Navigation
- Ensure all functions are accessible via keyboard
- Maintain a logical tab order
- Provide keyboard shortcuts for common actions

## 8. Testing

### 8.1 Manual Testing
- Test with different screen sizes
- Verify functionality with different browsers
- Test with both languages (English and Russian)

### 8.2 User Feedback
- Include feedback mechanisms in the UI
- Collect usage data to identify pain points
- Iterate based on user input