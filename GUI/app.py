from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel,QPushButton,QHBoxLayout,QApplication,QFileDialog,QSizePolicy,QFileDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QTimer,QSize
import sys
import cv2
import torch 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as tt
from torchvision.models import resnet50,ResNet50_Weights
from PIL import Image


from PyQt5.QtWidgets import QWidget

#----------------------------------------------------------------------------#

class MainAppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1280, 720)
        
        background_image = QPixmap(r'resources/background-1.png')
        if background_image.isNull():
            print("background.png not found. Using default background color.")
        else:
            background_label = QLabel(self)
            background_label.setPixmap(background_image)
        
      
        
        start_button = QPushButton('Get Started !', self)
        start_button.setStyleSheet('''
            QPushButton {
                background-color:#7898B6 ;
                color: white;
                border-radius: 7px;
                padding: 12px;
                font-family: arial;
                font-size: 25px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: grey;
            }
        ''')
        start_button.clicked.connect(self.nextwin)

        start_button.setGeometry(510,450,200,50)
        
        
        #self.setWindowIcon(QIcon("App_Icon.png"))
        self.setWindowTitle('PhokshoSathi')
        self.setWindowIcon(QIcon("appicon.ico"))
        self.showFullScreen()
        self.window2=None
  #----------------------------------------------------------------------------#
          
    def nextwin(self):
        self.close()
        if not self.window2:
            self.window2=View_Window()
            self.window2.show()
      
#----------------------------------------------------------------------------#
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        elif event.key() == Qt.Key_Escape:
            self.close()
        
#----------------------------------------------------------------------------#
            

class View_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1280, 720)
        background_image = QPixmap(r'resources/background-2.png')
        if background_image.isNull():
            print("background.png not found. Using default background color.")
        else:
            background_label = QLabel(self)
            background_label.setPixmap(background_image)

        # Set window properties
        self.setWindowTitle('PhokshoSathi')
        self.init_ui()
        self.showFullScreen()
        self.label2 = None
        self.label3=None
        self.labelnp=None

    def init_ui(self):
        # Title Label
        self.label = QLabel("PhokshoBot Requests You To Upload An Image", self)
        self.label.setStyleSheet('''
            QLabel {
                color: #ffffff;
                font-size: 35px;
                font-family: arial;
                font-weight: bold;
            }
        ''')
        self.label.setGeometry(200, 20, 850, 50)


        self.open_button = QPushButton("Select a Chest X-Ray image", self)
        self.open_button.setStyleSheet('''
            QPushButton {
                background-color: #2F2F2F;
                color: white;
                border-radius: 9px;
                padding: 28px;
                font-family: arial;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #D3D3D3;
            }
        ''')
        self.open_button.setGeometry(400, 150, 400, 400)
        self.open_button.clicked.connect(self.openFile)
        self.open_button.clicked.connect(self.proceed)
        

        self.home_button = QPushButton("Home", self)
        self.home_button.setStyleSheet('''
            QPushButton {
                background-color: #2F2F2F;
                color: white;
                border-radius: 9px;
                font-family: arial;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #D3D3D3;
            }
        ''')
        self.home_button.setGeometry(1130, 300, 100, 40) 
        self.home_button.clicked.connect(self.go_home)

        self.window2 = None

    def openFile(self):
        opt = QFileDialog.Options()
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        filepath, file = QFileDialog.getOpenFileName(self, "Open Image File", "", file_filter, options=opt)
        if filepath:
            file = cv2.imread(filepath)
            file2 = cv2.resize(file, (400, 400))
            cv2.imwrite("temp.png", file2)
            self.open_button.setIcon(QIcon("temp.png"))
            self.open_button.setIconSize(QSize(400, 400))
            self.open_button.setText("")
            transform = tt.Compose([
                tt.Resize(255),
                tt.CenterCrop(224),
               
                tt.ToTensor(),
                tt.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                tt.Lambda(lambda x: x[:3] if x.shape[0] == 4 else x) 
            ])
            image = Image.open(filepath)
            self.transformed_image = transform(image).unsqueeze(0)

    def proceed(self):
        probabilities = self.load_checkpoint("TrainedModel.pth", self.transformed_image)
        predicted_class = probabilities.argmax(dim=1).item()
        predicted_confidence = 100 / (1 + torch.exp(probabilities[0, predicted_class]))

        messages = [
            "No signs of pneumonia were detected.",
            "The scan indicates signs of pneumonia."
        ]
        messages_np= [
            "निमोनियाको कुनै लक्षण फेला परेन",
            "निमोनियाका लक्षणहरू भेटिए"
        ]
        confidence_text = f"Confidence: {100 - predicted_confidence:.2f}%"

        index = predicted_class
        if self.label2 is not None:
            self.label2.deleteLater()
            self.label2 = None
        if self.labelnp is not None:
            self.labelnp.deleteLater()
            self.labelnp = None
        if self.label3 is not None:
            self.label3.deleteLater()
            self.label3 = None

        self.label2 = QLabel(messages[index], self)
        self.labelnp=QLabel(messages_np[index],self)
        self.label3 = QLabel(confidence_text, self)

        self.label2.setStyleSheet(f'''
            QLabel {{
                color: {'#1fd655' if index == 0 else '#ee4b2b'};
                font-size: 25px;
                font-family: Arial;
                font-weight: bold;
            }}
        ''')
        self.labelnp.setStyleSheet(f'''
            QLabel {{
                color: {'#1fd655' if index == 0 else '#ee4b2b'};
                font-size: 25px;
                font-family: Arial;
                font-weight: bold;
            }}
        ''')

        self.label3.setStyleSheet('''
            QLabel {
                color: #FFFFFF;
                font-size: 25px;
                font-family: Arial;
                font-style: italic;
            }
        ''')

        self.label.setText("                        PhokshoBot's Diagnosis")
        self.label2.setGeometry(300, 580, 850, 50)
        self.labelnp.setGeometry(300, 620, 850, 50)
        self.label3.setGeometry(300, 660, 850, 50)

        self.label2.show()
        self.labelnp.show()
        self.label3.show()


    def load_checkpoint(self, filepath, image):
        device="cpu"
        model=PneumoniaClassifier()
        model.load_state_dict(torch.load(filepath))
        return model(image)


    def go_home(self):
        self.close()
        self.window2=MainAppWindow()
        self.window2.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        if event.key() == Qt.Key_Escape:
            self.close()



    
    


class PneumoniaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #Load the PreTrained Resnet50
        self.model=resnet50(weights=ResNet50_Weights.DEFAULT)
        
        for params in self.model.parameters():
            #Freeze all parameters
            params.require_grad=False

        #Create a new fully connected layer with unfrozen params
        self.model.fc=nn.Linear(self.model.fc.in_features,2)

    def forward(self,x):
        return self.model(x)
            
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#

if __name__ == "__main__":
    app = QApplication([])
    window = MainAppWindow()
    window.show()
    sys.exit(app.exec_())
