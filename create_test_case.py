import shutil

class CreateTestSet:
    def __init__(self):
        self.src_path = "dataset/train"
        self.dest_path = "dataset/test"

    def run(self):
        shutil.copytree(self.src_path, self.dest_path)

    def reset(self):
        try:
            shutil.rmtree(self.dest_path)
        except:
            pass