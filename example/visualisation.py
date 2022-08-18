# import time
# import napari
# from napari.qt.threading import thread_worker
# from qtpy.QtWidgets import QPushButton
# import numpy as np

# viewer = napari.Viewer()

# def update_layer(new_image):
#     try:
#         viewer.layers['result'].data = new_image
#     except KeyError:
#         viewer.add_image(
#             new_image, name='result', contrast_limits=(-0.8, 0.8)
#         )

# @thread_worker
# def yield_random_images_forever():
#     i = 0
#     while True:  # infinite loop!
#         yield np.random.rand(512, 512) * np.cos(i * 0.2)
#         i += 1
#         time.sleep(0.05)

# worker = yield_random_images_forever()
# worker.yielded.connect(update_layer)
# # add a button to the viewer that, when clicked, stops the worker

# button = QPushButton("START/STOP")
# button.clicked.connect(worker.toggle_pause)
# viewer.window.add_dock_widget(button)


# # on call updatae viewer

# # let other code run in background


# worker.start()
# napari.run()

# TODO: watchdog?

# import napari
# import time

# from napari.qt.threading import thread_worker
# from qtpy.QtWidgets import QLineEdit, QLabel, QWidget, QVBoxLayout
# from qtpy.QtGui import QDoubleValidator


# @thread_worker
# def multiplier():
#     total = 1
#     while True:
#         time.sleep(0.1)
#         new = yield total
#         total *= new if new is not None else 1
#         if total == 0:
#             return "Game Over!"

# viewer = napari.Viewer()

# # make a widget to control the worker
# # (not the main point of this example...)
# widget = QWidget()
# layout = QVBoxLayout()
# widget.setLayout(layout)
# result_label = QLabel()
# line_edit = QLineEdit()
# line_edit.setValidator(QDoubleValidator())
# layout.addWidget(line_edit)
# layout.addWidget(result_label)
# viewer.window.add_dock_widget(widget)

# # create the worker
# worker = multiplier()

# # define some callbacks
# def on_yielded(value):
#     worker.pause()
#     result_label.setText(str(value))
#     line_edit.setText('1')

# def on_return(value):
#     line_edit.setText('')
#     line_edit.setEnabled(False)
#     result_label.setText(value)

# def send_next_value():
#     worker.send(float(line_edit.text()))
#     worker.resume()

# worker.yielded.connect(on_yielded)
# worker.returned.connect(on_return)
# line_edit.returnPressed.connect(send_next_value)

# worker.start()
# napari.run()


import numpy as np
import napari
from qtpy.QtCore import QTimer

viewer = napari.Viewer()

def save_stuff():
    # this will depend very much on the contents of viewer.layers
    # so you may want to refine this function.
    if viewer.layers:
        print("layer update")
        # viewer.layers.refresh()
        # viewer.layers.save('~/Desktop/layers')

viewer.add_image(np.random.random((500, 500)))

timer = QTimer()
timer.timeout.connect(save_stuff)
timer.start(1000)  # fire every 1 second

napari.run()


# napari has to be the main
# https://forum.image.sc/t/use-napari-in-a-separate-thread/65372/2
# https://github.com/haesleinhuepf/napari-image-quality-analyzer/blob/master/napari_image_quality_analyzer/_dock_widget.py
# https://github.com/BiAPoL/on_the_fly_image_processing_napari/blob/680d9ceeef5ae188541a96c7125f0fca07f28af5/05_on_the_fly_processing.py
# https://f1000research.com/slides/10-390