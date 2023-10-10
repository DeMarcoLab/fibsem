import napari
from qtdesigner_files import image_viewer
from PyQt5 import QtWidgets
from datetime import datetime
from pathlib import Path

class image_viewer(image_viewer.Ui_MainWindow,QtWidgets.QMainWindow):

    def __init__(self, viewer: napari.Viewer):
        super(image_viewer, self).__init__()
        self.setupUi(self)

        self.setMinimumWidth(400)
        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(True)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.image = None

        # signal when image is dragged and dropped into viewer
        self.viewer.layers.events.inserted.connect(self.update_layer_list)

        self.viewer.layers.selection.events.changed.connect(self.update_selected_layer)

        self.pushButton_export.clicked.connect(self.export_metadata)

    def export_metadata(self):
        from fibsem.utils import _display_metadata
        import matplotlib.pyplot as plt
        if self.image is None: 
            napari.utils.notifications.show_info("No image loaded")
            return
        
        path = self._get_save_file_ui()

        fig = _display_metadata(self.image,show=False)

        fig.savefig(path)

        napari.utils.notifications.show_info(f"Image Saved with metadata")


    def _get_save_file_ui( msg: str = "Select a file",_filter: str = "*png",) -> Path:
   
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            msg,
            filter=_filter,
        )
        return path

    def update_layer_list(self, event):

        for layer in self.viewer.layers:
            print(layer.name)

        layer1 = self.viewer.layers[0]


    def update_selected_layer(self,event):

        from fibsem.structures import FibsemImage, FibsemImageMetadata
        from fibsem import constants


        selected_layer = self.viewer.layers.selection.active

        if selected_layer is None:
            return

        for layer in self.viewer.layers:
            
            layer.visible = False

        selected_layer.visible = True


        self.image = FibsemImage.load(selected_layer.source.path)

        image_settings = self.image.metadata.image_settings

        beam_type = image_settings.beam_type.name
        resolution = f'{image_settings.resolution[0]} x {image_settings.resolution[1]}'
        hfw = image_settings.hfw * constants.SI_TO_MICRO
        name = image_settings.label

        self.label_beamType.setText(beam_type)
        self.label_resolution.setText(resolution)
        self.label_hfw.setText(f'{hfw:.2f}')
        self.label_name.setText(name)
        self.label_metadata_version.setText(self.image.metadata.version)

        timestamp = datetime.fromtimestamp(self.image.metadata.microscope_state.timestamp)
        self.label_timestamp.setText(timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        experiment_data = self.image.metadata.experiment
        self._load_experiment_info(experiment_data)

        position = self.image.metadata.microscope_state.absolute_position

        self.label_position.setText(f'x: {position.x:.2f} y: {position.y:.2f} z: {position.z:.2f} r: {position.r:.2f} t: {position.t:.2f}')

        if beam_type == "ION":

            detector_data = self.image.metadata.microscope_state.ib_detector
            beam_data = self.image.metadata.microscope_state.ib_settings
        
        else:

            detector_data = self.image.metadata.microscope_state.eb_detector
            beam_data = self.image.metadata.microscope_state.eb_settings

        self._load_detector_info(detector_data)

        self._load_beam_info(beam_data)

        self._load_system_info(self.image.metadata.system)

        self._load_user_info(self.image.metadata.user)



    def _load_experiment_info(self,experiment):

        if experiment is None:
            text = "No Information"
            self.label_ID.setText(text)
            self.label_method.setText(text)
            self.label_application.setText(text)
            self.label_app_version.setText("")
            self.label_fibsem_version.setText(text)

        else:
            self.label_ID.setText(experiment.id if experiment.id is not None else "")
            self.label_method.setText(experiment.method if experiment.method is not None else "")
            self.label_application.setText(experiment.application if experiment.application is not None else "")
            self.label_app_version.setText(experiment.application_version   if experiment.application_version is not None else "")
            self.label_fibsem_version.setText(experiment.fibsem_version if experiment.fibsem_version is not None else "")


    def _load_detector_info(self,detector):

        if detector is None:
            text = "No Information"
            self.label_detector_type.setText(text)
            self.label_detector_mode.setText(text)
            self.label_detector_brightness.setText(text)
            self.label_detector_contrast.setText(text)

        else:

            self.label_detector_type.setText(detector.type if detector.type is not None else "")
            self.label_detector_mode.setText(detector.mode if detector.mode is not None else "")
            self.label_detector_brightness.setText(f'{detector.brightness:.2f}' if detector.brightness is not None else "")
            self.label_detector_contrast.setText(f'{detector.contrast:.2f}' if detector.contrast is not None else "")
        
        

    def _load_beam_info(self,beam):

        if beam is None:
            text = "No Information"
            self.label_beam_current.setText(text)
            self.label_voltage.setText(text)
            self.label_working_distance.setText(text)
            self.label_dwell_time.setText(text)
            self.label_scan_rotation.setText(text)
            self.label_shift.setText(text)
            self.label_stigmation.setText(text)

        else:

            self.label_beam_current.setText(f'{beam.beam_current:.2f}' if beam.beam_current is not None else "")
            self.label_voltage.setText(f'{beam.voltage:.2f}' if beam.voltage is not None else "")
            self.label_working_distance.setText(f'{beam.working_distance:.2f}' if beam.working_distance is not None else "")
            self.label_dwell_time.setText(f'{beam.dwell_time:.2f}' if beam.dwell_time is not None else "")
            self.label_scan_rotation.setText(f'{beam.scan_rotation:.2f}' if beam.scan_rotation is not None else "")

            shift_text = f'x: {beam.shift.x:.2f} y: {beam.shift.y:.2f}' if beam.shift is not None else ""
            
            self.label_shift.setText(shift_text)

            stigmation_text = f'x: {beam.stigmation.x:.2f} y: {beam.stigmation.y:.2f}' if beam.stigmation is not None else ''

            self.label_stigmation.setText(stigmation_text)





    def _load_system_info(self,system):

        if system is None:
            text = "No Information"
            self.label_manufacturer.setText(text)
            self.label_model.setText(text)
            self.label_serial_no.setText(text)
            self.label_software_version.setText(text)

        else:

            self.label_manufacturer.setText(system.manufacturer if system.manufacturer is not None else "")
            self.label_model.setText(system.model if system.model is not None else "")
            self.label_serial_no.setText(system.serial_number if system.serial_number is not None else "")
            self.label_software_version.setText(system.software_version if system.software_version is not None else "")

    def _load_user_info(self,user):

        if user is None:
            text = "No Information"
            self.label_username.setText(text)
            self.label_email.setText(text)
            self.label_organisation.setText(text)
            self.label_userpc.setText(text)

        else:

            self.label_username.setText(user.name if user.name is not None else "")
            self.label_email.setText(user.email if user.email is not None else "")
            self.label_organisation.setText(user.organization if user.organization is not None else "")
            self.label_userpc.setText(user.computer if user.computer is not None else "")


        





def main():
    pass


if __name__ == "__main__":
    main()