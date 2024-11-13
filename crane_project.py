import yaml
import os
import cv2
import csv
import logging
from ultralytics import YOLO
def training(epoch):
    """
        The training function encapsulates the whole process of setting up the environment,
        reading configurations, initializing, and training the YOLO model.
        """

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class EnvironmentManager:
        """
        A class used to manage environment variables.

        ...

        Attributes
        ----------
        env_vars : dict
            a dictionary holding environment variable names and their values

        Methods
        -------
        set_env_vars():
            Sets the environment variables defined in env_vars.
        """

        def __init__(self, env_vars):
            self.env_vars = env_vars

        def set_env_vars(self):
            for var, value in self.env_vars.items():
                os.environ[var] = value
                logger.info(f"Environment variable {var} set to {value}.")

    class ConfigManager:
        """
        A class used to manage configuration files.

        ...

        Attributes
        ----------
        config_path : str
            a string representing the path to the configuration file

        Methods
        -------
        read_yaml():
            Reads the YAML configuration file and returns the data.
        """

        def __init__(self, config_path):
            self.config_path = config_path

        def read_yaml(self):
            """
            Reads the YAML configuration file and returns the data.

            Returns
            -------
            dict
                a dictionary representing the YAML data
            """
            try:
                with open(self.config_path, 'r') as stream:
                    data = yaml.safe_load(stream)
                    logger.info(f"YAML Data: {data}")
                    return data
            except FileNotFoundError:
                logger.error(f"{self.config_path} not found.")
                raise
            except yaml.YAMLError as exc:
                logger.error(f"Error occurred while reading YAML: {exc}")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred while reading YAML: {e}")
                raise

    class DirectoryManager:
        """
        A class used to manage directory-related tasks.

        ...

        Methods
        -------
        get_all_files(directory: str) -> list:
            Gets all files from the specified directory.
        """

        @staticmethod
        def get_all_files(directory: str) -> list:
            try:
                files = [os.path.join(directory, f) for f in os.listdir(directory) if
                         os.path.isfile(os.path.join(directory, f))]
                logger.info(f"All files from {directory} have been loaded.")
                return files
            except FileNotFoundError:
                logger.error(f"{directory} not found.")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred while getting all files from {directory}: {e}")
                raise

    class FileManager:
        """
        A class used to manage file-related tasks.

        ...

        Methods
        -------
        read_text_file(file_path: str) -> str:
            Reads the content of a text file and returns it.
        """

        @staticmethod
        def read_text_file(file_path: str) -> str:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    logger.info(f"Content from {file_path} has been read.")
                    return content
            except FileNotFoundError:
                logger.error(f"{file_path} not found.")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
                raise

    class YOLOTrainer:
        """
        A class used to initialize and train YOLO models.

        ...

        Attributes
        ----------
        model_config : str
            a string representing the path to the model configuration file
        data_config : str
            a string representing the path to the data configuration file
        model : YOLO
            an instance of the YOLO model

        Methods
        -------
        initialize_model():
            Initializes the YOLO model.
        train_model():
            Trains the YOLO model with the specified parameters.
        """

        def initialize_model(self):
            """Initializes and logs the YOLO model."""
            try:
                self.model = YOLO(self.model_config)
                logger.info("Model initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing model: {e}")
                raise


        def load_images(self, img_dir: str):
            """
            Loads all images from the specified directory.

            Parameters
            ----------
            img_dir : str
                a string representing the path to the image directory
            """
            try:
                self.images = DirectoryManager.get_all_files(img_dir)
                logger.info("Images loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading images: {e}")
                raise

        def read_additional_configs(self, config_dir: str):
            """
            Reads additional configuration files from the specified directory.

            Parameters
            ----------
            config_dir : str
                a string representing the path to the configuration directory
            """
            try:
                config_files = DirectoryManager.get_all_files(config_dir)
                self.additional_configs = [FileManager.read_text_file(file) for file in config_files]
                logger.info("Additional configurations read successfully.")
            except Exception as e:
                logger.error(f"Error reading additional configurations: {e}")
                raise

        def __init__(self, model_config, data_config):
            self.model_config = model_config
            self.data_config = data_config
            self.model = None



        def train_model(self, yaml_data):
            """
            Trains the YOLO model with the specified parameters.

            Parameters
            ----------
            yaml_data : dict
                a dictionary representing the YAML data
            """
            try:
                results = self.model.train(
                    task="detect",
                    mode="train",
                    imgsz=640,
                    device=0,
                    data=self.data_config,
                    epochs=epoch,
                    optimizer='Adam',
                    iou=0.7,
                    batch=4
                )
                logger.info(f"Training Results: {results}")
            except Exception as e:
                logger.error(f"Error occurred during training: {e}")
                raise

    def main():
        """
        The main function to set environment variables, read configurations, initialize, and train the model.
        """
        # Set Environment Variables
        env_manager = EnvironmentManager({"KMP_DUPLICATE_LIB_OK": "TRUE"})
        env_manager.set_env_vars()

        # Read and Print YAML Data
        config_manager = ConfigManager("config.yaml")
        yaml_data = config_manager.read_yaml()

        # Initialize and Train YOLO Model
        trainer = YOLOTrainer("yolov8x.yaml", "config.yaml")
        trainer.initialize_model()
        trainer.train_model(yaml_data)


    try:
        main()
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}")

def testing():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class TestEnvironmentManager:
        """
        A class used to manage the testing environment.

        ...

        Attributes
        ----------
        test_dir : str
            a string representing the path to the test directory
        output_dir : str
            a string representing the path to the output directory
        csv_file_path : str
            a string representing the path to the CSV file

        Methods
        -------
        setup_environment():
            Sets up the testing environment by creating necessary directories.
        """

        def __init__(self, test_dir, output_dir, csv_file_path):
            self.test_dir = test_dir
            self.output_dir = output_dir
            self.csv_file_path = csv_file_path

        def setup_environment(self):
            """Sets up the testing environment by creating necessary directories."""
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                logger.info(f"Created output directory: {self.output_dir}")
            else:
                logger.info(f"Output directory already exists: {self.output_dir}")

            if not os.path.exists(self.test_dir):
                logger.error(f"Test directory does not exist: {self.test_dir}")
                raise FileNotFoundError(f"Test directory does not exist: {self.test_dir}")

            if not os.path.exists(self.csv_file_path):
                with open(self.csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['image', 'label', 'x_min', 'y_min', 'x_max', 'y_max'])
                logger.info(f"Created CSV file: {self.csv_file_path}")
            else:
                logger.info(f"CSV file already exists: {self.csv_file_path}")

    class DirectoryManager:
        """
        A class used to manage directory-related tasks.

        ...

        Methods
        -------
        get_latest_train_folder(base_path: str) -> str:
            Gets the latest train folder from the base path.
        load_images_from_dir(directory: str) -> list:
            Loads all images from the specified directory.
        """

        @staticmethod
        def get_latest_train_folder(base_path: str) -> str:
            train_folders = [f for f in os.listdir(base_path) if f.startswith('train')]
            train_folders = sorted(train_folders, key=lambda x: os.path.getctime(os.path.join(base_path, x)),
                                   reverse=True)
            return train_folders[0] if train_folders else None

        @staticmethod
        def load_images_from_dir(directory: str) -> list:
            images = []
            for filename in os.listdir(directory):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(directory, filename)
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        images.append((filename, frame))
                    else:
                        logger.error(f"Error reading image: {img_path}")
            return images

    class FileManager:
        """
        A class used to manage file-related tasks.

        ...

        Methods
        -------
        create_csv_file(csv_file_path: str):
            Creates a new CSV file with specified headers.
        write_to_csv(csv_file_path: str, row: list):
            Writes a row to the specified CSV file.
        """

        @staticmethod
        def create_csv_file(csv_file_path: str):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['image', 'label', 'x_min', 'y_min', 'x_max', 'y_max'])

        @staticmethod
        def write_to_csv(csv_file_path: str, row: list):
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)

    class ImageManager:
        """
        A class used to manage image-related tasks.

        ...

        Methods
        -------
        save_image(output_dir: str, filename: str, frame: object):
            Saves the image to the specified output directory.
        """

        @staticmethod
        def save_image(output_dir: str, filename: str, frame: object):
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            logger.info(f"Output written to {output_path}")

    class YOLOTester:
        """
        A class used to test YOLO models.

        ...

        Attributes
        ----------
        model : YOLO
            an instance of the YOLO model
        threshold : float
            a float representing the detection threshold

        Methods
        -------
        load_model(model_path: str):
            Loads the YOLO model from the specified path.
        run_tests(test_dir: str, output_dir: str, csv_file_path: str):
            Runs tests on the images in the test directory and writes the results to the output directory and CSV file.
        """

        def __init__(self, threshold=0.5):
            self.model = None
            self.threshold = threshold

        def load_model(self, model_path: str):
            """Loads the YOLO model from the specified path."""
            try:
                self.model = YOLO(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
                raise

        def run_tests(self, test_dir: str, output_dir: str, csv_file_path: str):
            """
            Runs tests on the images in the test directory and writes the results to the output directory and CSV file.

            Parameters:
            test_dir (str): The directory containing the test images.
            output_dir (str): The directory where the output images will be saved.
            csv_file_path (str): The path of the CSV file where the results will be written.
            """
            try:
                # Load images from the test directory
                images = DirectoryManager.load_images_from_dir(test_dir)

                # Iterate over each image and its filename
                for filename, frame in images:
                    height, width, _ = frame.shape
                    results = self.model(frame)[0]

                    # Initialize detected classes dictionary
                    detected_classes = {}
                    for class_name in ["PERSON", "CRANE", "HOOK", "CRANE_ARM"]:
                        detected_classes[class_name] = []

                    # Process each detection result
                    for result in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = result
                        if score > self.threshold:
                            class_name = results.names[int(class_id)].upper()
                            detected_classes[class_name].append((x1, y1, x2, y2))
                            FileManager.write_to_csv(csv_file_path, [os.path.basename(filename), class_id, x1, y1, x2, y2])


                    # Initialize condition and color
                    condition = "Not Detected"
                    condition_color = (255, 0, 0)

                    # Check conditions and update accordingly
                    if "PERSON" in detected_classes:
                        for person_box in detected_classes["PERSON"]:
                            for other_class in ["CRANE", "HOOK", "CRANE_ARM"]:
                                if other_class in detected_classes:
                                    for other_box in detected_classes[other_class]:
                                        if (person_box[0] < other_box[2] and person_box[2] > other_box[0] and
                                                person_box[1] < other_box[3] and person_box[3] > other_box[1]):
                                            condition = "Unsafe"
                                            condition_color = (0, 0, 255)
                                            break

                    if condition != "Unsafe" and any(
                            class_name in detected_classes for class_name in ["PERSON", "CRANE", "HOOK", "CRANE_ARM"]):
                        condition = "Safe"
                        condition_color = (0, 255, 0)

                    # Draw bounding boxes and put text on the image
                    for result in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = result
                        if score > self.threshold:
                            class_name = results.names[int(class_id)].upper()
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), condition_color, 4)
                            cv2.putText(frame, f"{class_name} {score * 100:.2f}%", (int(x1), int(y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, condition_color, 2, cv2.LINE_AA)

                    # Put condition text on the image
                    text_size = cv2.getTextSize(condition, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = 50
                    cv2.putText(frame, condition, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, condition_color, 2,
                                cv2.LINE_AA)

                    # Save the output image
                    ImageManager.save_image(output_dir, filename, frame)

            except Exception as e:
                logger.error(f"An error occurred while running tests: {e}")
                raise

    def get_latest_train_folder(base_path):
        """
        Gets the latest train folder from the base path.

        Parameters
        ----------
        base_path : str
            a string representing the base path where train folders are located

        Returns
        -------
        str
            a string representing the name of the latest train folder
        """
        train_folders = [f for f in os.listdir(base_path) if f.startswith('train')]
        train_folders = sorted(train_folders, key=lambda x: os.path.getctime(os.path.join(base_path, x)), reverse=True)
        return train_folders[0] if train_folders else None

    def main_code1():
        """
        The main function to setup the environment, load the model, and run tests.
        """
        base_path = os.path.join('.', 'runs', 'detect')
        latest_train_folder = get_latest_train_folder(base_path)
        if latest_train_folder is None:
            logger.error("No train folder found!")
            exit(1)

        logger.info(f"Using the latest train folder: {latest_train_folder}")

        model_path = os.path.join(base_path, latest_train_folder, 'weights', 'last.pt')
        test_dir = '/home/nicku/PycharmProjects/crane_project/dataset/test_dataset/test'
        output_dir = '/home/nicku/PycharmProjects/crane_project/dataset/test_dataset/pred'
        csv_file_path = 'output.csv'

        env_manager = TestEnvironmentManager(test_dir, output_dir, csv_file_path)
        env_manager.setup_environment()

        tester = YOLOTester()
        tester.load_model(model_path)
        tester.run_tests(test_dir, output_dir, csv_file_path)

    try:
        main_code1()
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}")

if __name__ == '__main__':
    training(epoch = 10)
    testing()
