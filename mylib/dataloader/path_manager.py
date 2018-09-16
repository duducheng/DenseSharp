import os


class PathManager:
    def __init__(self, base):
        self.base = base

    @property
    def raw_data(self):
        return os.path.join(self.base, "raw")

    @property
    def raw_info(self):
        return os.path.join(self.raw_data, "pathologic-20171110.xlsx")

    @property
    def raw2_data(self):
        return os.path.join(self.base, "raw2")

    @property
    def raw2_info(self):
        return os.path.join(self.raw2_data, "10mm_2nd.xlsx")

    @property
    def raw3_data(self):
        return os.path.join(self.base, "raw3")

    @property
    def raw3_info(self):
        return os.path.join(self.raw3_data, "pathologic_name3rd.xlsx")

    @property
    def processed(self):
        return os.path.join(self.base, "processed")

    @property
    def case_data(self):
        return os.path.join(self.processed, "case")

    @property
    def case_info(self):
        return os.path.join(self.processed, "info.csv")

    @property
    def nodule_data(self):
        return os.path.join(self.processed, 'nodule')

    def get_patient_folder(self, patient):
        if patient.startswith("f"):  # first set
            return os.path.join(self.raw_data, "patient1 ({patient})".format(patient=patient[1:]))
        if patient.startswith("s"):  # second set
            return os.path.join(self.raw2_data, "0 ({patient})".format(patient=patient[1:]))
        if patient.startswith("t"):  # second set
            return os.path.join(self.raw3_data, "1 ({patient})".format(patient=patient[1:]))
        raise ValueError("There are only `first (f)`, `second (s)` and `third (t)`.")

    def get_patient_case(self, patient):
        return os.path.join(self.case_data, "{patient}.npz".format(patient=patient))

    def get_nodule(self, patient):
        return os.path.join(self.nodule_data, "{patient}.npz".format(patient=patient))


DATASET_BASE = "HD_DATASET"
if DATASET_BASE in os.environ:
    PATH = PathManager(os.environ[DATASET_BASE])  # singleton
else:
    print("Cannot setup `PATH` singleton.")
    PATH = None
