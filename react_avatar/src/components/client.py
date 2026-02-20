from PyQt5.QtCore import QUrl, Qt, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile, QWebEnginePage, QWebEngineSettings
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import sys

RETRY_INTERVAL_MS = 5000  # retry every 5 seconds
AVATAR_URL = "http://192.168.0.171:5173/"

class AvatarApp:
    def __init__(self):
        self.app = QApplication(sys.argv)

        # Loading screen setup
        self.loading_window = QWidget()
        self.loading_window.setWindowTitle("Hologram AI")
        self.loading_window.setStyleSheet("background-color: black; color: white;")
        layout = QVBoxLayout()
        self.label = QLabel("Avatar loading...")
        self.label.setStyleSheet("font-size: 32px; font-weight: bold; color: white;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.loading_window.setLayout(layout)
        self.loading_window.setGeometry(300, 300, 600, 200)
        self.loading_window.show()

        # Webview setup
        self.webview = QWebEngineView()
        profile = QWebEngineProfile("my_profile", self.webview)
        profile.defaultProfile().setPersistentCookiesPolicy(QWebEngineProfile.ForcePersistentCookies)

        self.webpage = QWebEnginePage(profile, self.webview)
        self.webpage.settings().setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)
        self.webview.setPage(self.webpage)

        # Retry timer
        self.retry_timer = QTimer()
        self.retry_timer.setInterval(RETRY_INTERVAL_MS)
        self.retry_timer.timeout.connect(self.load_avatar)

        # Page load signal
        self.webview.loadFinished.connect(self.on_load_finished)

        # Start initial load
        self.load_avatar()

    def load_avatar(self):
        self.label.setText("Avatar loading...")
        self.webview.load(QUrl(AVATAR_URL))

    def on_load_finished(self, success):
        if success:
            self.retry_timer.stop()
            self.loading_window.close()
            self.webview.showFullScreen()
        else:
            self.label.setText("Failed to load avatar.\nRetrying in 5 seconds...")
            self.retry_timer.start()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    AvatarApp().run()
