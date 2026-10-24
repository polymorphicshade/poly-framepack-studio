progress_html = '''
<div class="loader-container">
  <div class="loader"></div>
  <div class="progress-container">
    <progress value="*number*" max="100"></progress>
  </div>
  <span>*text*</span>
</div>
'''

css = '''
/* poly-wan's progress row: a small spinner, a thin accent bar and muted text.
   Colours come from the Gradio theme variables, so it follows light and dark mode. */
.loader-container {
  display: flex;
  align-items: center;
  gap: 10px;
  white-space: nowrap;
  font-size: 13px;
  color: var(--body-text-color-subdued);
}

.loader {
  border: 2px solid var(--border-color-primary);
  border-top: 2px solid var(--color-accent);
  border-radius: 50%;
  width: 14px;
  height: 14px;
  flex-shrink: 0;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

progress {
  appearance: none;
  -webkit-appearance: none;
  border: none;
  height: 6px;
  border-radius: 3px;
  overflow: hidden;
  background-color: var(--background-fill-secondary);
  width: 100%;
  vertical-align: middle !important;
}

.progress-container {
  flex-grow: 1;
}

progress::-webkit-progress-bar {
  background-color: var(--background-fill-secondary);
}

progress::-webkit-progress-value {
  background-color: var(--color-accent);
  transition: width .2s;
}

progress::-moz-progress-bar {
  background-color: var(--color-accent);
}

.no-generating-animation > .generating {
  display: none !important;
}

'''


def make_progress_bar_html(number, text):
    return progress_html.replace('*number*', str(number)).replace('*text*', text)


def make_progress_bar_css():
    return css
