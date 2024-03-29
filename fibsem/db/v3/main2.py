import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

class UserCreationForm(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Create New User')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Username
        self.usernameLabel = QLabel('Username:')
        self.usernameLineEdit = QLineEdit()
        layout.addWidget(self.usernameLabel)
        layout.addWidget(self.usernameLineEdit)

        # Name
        self.nameLabel = QLabel('Name:')
        self.nameLineEdit = QLineEdit()
        layout.addWidget(self.nameLabel)
        layout.addWidget(self.nameLineEdit)

        # Email
        self.emailLabel = QLabel('Email:')
        self.emailLineEdit = QLineEdit()
        layout.addWidget(self.emailLabel)
        layout.addWidget(self.emailLineEdit)

        # Password
        self.passwordLabel = QLabel('Password:')
        self.passwordLineEdit = QLineEdit()
        self.passwordLineEdit.setEchoMode(QLineEdit.Password)  # Hide password input
        layout.addWidget(self.passwordLabel)
        layout.addWidget(self.passwordLineEdit)

        # Role
        self.roleLabel = QLabel('Role:')
        self.roleLineEdit = QLineEdit()
        layout.addWidget(self.roleLabel)
        layout.addWidget(self.roleLineEdit)

        # Submit Button
        self.submitButton = QPushButton('Submit')
        self.submitButton.clicked.connect(self.submitForm)
        layout.addWidget(self.submitButton)

        self.setLayout(layout)

    def submitForm(self):
        # Retrieve the input values
        username = self.usernameLineEdit.text()
        name = self.nameLineEdit.text()
        email = self.emailLineEdit.text()
        password = self.passwordLineEdit.text()
        role = self.roleLineEdit.text()

        # For demonstration, print the values. Replace this with your actual form submission logic.
        print(f"Username: {username}, Name: {name}, Email: {email}, Password: {password}, Role: {role}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UserCreationForm()
    ex.show()
    sys.exit(app.exec_())
