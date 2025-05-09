import atexit
import time
from selenium import webdriver


url = 'http://127.0.0.1:8000'

def _safe_quit(browser):
    """Best‑effort shutdown that won’t crash if the driver is already gone."""
    try:
        browser.quit()
    except Exception:
        pass

def open_browser(url=url, headless=False):

    # Set up Selenium WebDriver
    options = webdriver.ChromeOptions()
    if headless:

        # Enable browser console logging
        options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    else:
        options.add_argument("--auto-open-devtools-for-tabs")  # Open developer tools
        options.add_argument("--start-maximized")  # Open browser maximized

    driver = webdriver.Chrome(options=options)
    driver.get(url)  # Open the URL

    # Switch to the "console" tab of the developer tools
    # driver.execute_script("window.open('');")
    # driver.switch_to.window(driver.window_handles[1])
    # driver.get('chrome://devtools/console')
    atexit.register(_safe_quit, driver)

    seen = set()
    print("Streaming JS console output...\n")

    try:
        while True:
            logs = driver.get_log("browser")
            for entry in logs:
                key = (entry['message'], entry['timestamp'])
                if key not in seen:
                    seen.add(key)
                    print(f"[{entry['level']}] {entry['message']}")
            time.sleep(0.5)  # Poll every 500ms
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        driver.quit()    

    return driver


if __name__ == "__main__":
    # Open the browser and navigate to the URL
    open_browser(url)

    # Register a function to close the browser when the script ends
