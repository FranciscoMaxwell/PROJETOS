import sys
from PyQt5.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView, 
    QGraphicsRectItem, QGraphicsTextItem, QInputDialog, QPushButton, QVBoxLayout, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
import svgwrite
import ezdxf
from shapely.geometry import box, Polygon

# ---------- Cômodo Interativo ----------
class RoomItem(QGraphicsRectItem):
    def __init__(self, name, x, y, w, h, color=None):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.name = name
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable | 
            QGraphicsRectItem.ItemIsSelectable | 
            QGraphicsRectItem.ItemSendsGaeometryChanges
        )
        self.setBrush(QBrush(color if color else QColor(100, 150, 200, 150)))
        # Label
        self.label = QGraphicsTextItem(name, self)
        self.update_label()

    def update_label(self):
        rect = self.rect()
        self.label.setPos(rect.width()/2 - self.label.boundingRect().width()/2,
                          rect.height()/2 - self.label.boundingRect().height()/2)

    def mouseDoubleClickEvent(self, event):
        # Renomear cômodo com duplo clique
        new_name, ok = QInputDialog.getText(None, "Renomear Cômodo", "Nome:", text=self.name)
        if ok and new_name.strip():
            self.name = new_name
            self.label.setPlainText(new_name)
            self.update_label()
        super().mouseDoubleClickEvent(event)

# ---------- Export SVG ----------
def export_svg(scene, filename="planta.svg"):
    # Detectar itens
    dwg = svgwrite.Drawing(filename, size=("1000px","800px"))
    for item in scene.items():
        if isinstance(item, RoomItem):
            x, y = item.pos().x(), item.pos().y()
            w, h = item.rect().width(), item.rect().height()
            dwg.add(dwg.rect(insert=(x, y), size=(w, h), fill='#69c', stroke='black', stroke_width=2))
            dwg.add(dwg.text(item.name, insert=(x + w/2, y + h/2),
                             text_anchor="middle", alignment_baseline="middle", font_size=20))
    dwg.save()
    print(f"[SVG] Salvo em {filename}")

# ---------- Export DXF ----------
def export_dxf(scene, filename="planta.dxf"):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    for item in scene.items():
        if isinstance(item, RoomItem):
            x, y = item.pos().x(), item.pos().y()
            w, h = item.rect().width(), item.rect().height()
            msp.add_lwpolyline([(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)], dxfattribs={'closed':True})
            cx, cy = x + w/2, y + h/2
            msp.add_text(item.name, dxfattribs={'height':0.3}).set_placement((cx, cy))
    doc.saveas(filename)
    print(f"[DXF] Salvo em {filename}")

# ---------- Aplicativo Principal ----------
class BlueprintApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blueprint Interativo")
        self.setGeometry(100,100,1200,800)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Cena e View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        # Botões export
        btn_svg = QPushButton("Exportar SVG")
        btn_svg.clicked.connect(lambda: export_svg(self.scene))
        layout.addWidget(btn_svg)

        btn_dxf = QPushButton("Exportar DXF")
        btn_dxf.clicked.connect(lambda: export_dxf(self.scene))
        layout.addWidget(btn_dxf)

        # Criar cômodos de exemplo
        self.add_example_rooms()

    def add_example_rooms(self):
        rooms = [
            ("Sala", 50, 50, 300, 200),
            ("Cozinha", 400, 50, 200, 150),
            ("Quarto1", 50, 300, 200, 150),
            ("Quarto2", 300, 300, 200, 150),
            ("Banheiro", 550, 300, 100, 100)
        ]
        colors = [QColor(255,0,0,150), QColor(0,255,0,150), QColor(0,0,255,150),
                  QColor(255,255,0,150), QColor(0,255,255,150)]
        for (name, x, y, w, h), color in zip(rooms, colors):
            self.scene.addItem(RoomItem(name, x, y, w, h, color))

# ---------- Rodar App ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlueprintApp()
    window.show()
    sys.exit(app.exec_())
