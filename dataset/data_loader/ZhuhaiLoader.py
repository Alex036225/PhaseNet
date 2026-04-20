"""The dataloader for Zhuhai Hospital dataset (Mindray monitor data).

This dataset contains:
- RGB video (Output_RGB.mkv)
- IR videos (Output_IR1.mkv, Output_IR2.mkv)
- Mindray monitor data (recorded_mindray.txt) in HL7 format
- Timestamp log (timestamp_log.txt)

The Mindray data contains:
- ECG signals (MDC_ECG_ELEC_POTL_I, II, III, AVR, AVL, AVF, V) at 500Hz
- PPG/SpO2 signal (MDC_PULS_OXIM_PLETH) at 60Hz
- Respiratory signal (MDC_IMPED_TTHOR) at 256Hz
"""
import glob
import os
import re
from datetime import datetime

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class ZhuhaiLoader(BaseLoader):
    """The data loader for the Zhuhai Hospital dataset."""

    def __init__(self, name, data_path, config_data, is_multi=True):
        """Initializes a Zhuhai dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and monitor data.
                e.g. data_path should be "/public_hw/share/cit_ztyu/zhaobo/zhuhai/save" for below dataset structure:
                -----------------
                     save/
                     |   |-- 12/
                     |       |-- 1/
                     |           |-- Output_RGB.mkv
                     |           |-- Output_IR1.mkv
                     |           |-- Output_IR2.mkv
                     |           |-- recorded_mindray.txt
                     |           |-- timestamp_log.txt
                     |   |-- 13/
                     |       |-- 1/
                     |       |-- 2/
                     |...
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, is_multi)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (For Zhuhai dataset).

        Supports two common layouts:
        1) data_path/patient_id/session_id/{Output_RGB.mkv, recorded_mindray.txt}
        2) data_path/subjectXX/{Output_RGB.mkv, recorded_mindray.txt}
        """
        data_dirs = []

        def _try_patient_session_layout():
            out = []
            patient_dirs = sorted(glob.glob(os.path.join(data_path, "*")))
            for patient_dir in patient_dirs:
                if not os.path.isdir(patient_dir):
                    continue
                patient_id = os.path.basename(patient_dir)

                session_dirs = sorted(glob.glob(os.path.join(patient_dir, "*")))
                for session_dir in session_dirs:
                    if not os.path.isdir(session_dir):
                        continue
                    session_id = os.path.basename(session_dir)

                    rgb_video = os.path.join(session_dir, "Output_RGB.mkv")
                    mindray_file = os.path.join(session_dir, "recorded_mindray.txt")

                    if os.path.exists(rgb_video) and os.path.exists(mindray_file):
                        index = f"{patient_id}_{session_id}"
                        try:
                            subject = int(patient_id)
                        except ValueError:
                            m = re.search(r"(\d+)", patient_id)
                            subject = int(m.group(1)) if m else -1
                        out.append(
                            {
                                "index": index,
                                "path": session_dir,
                                "subject": subject,
                                "patient_id": patient_id,
                                "session_id": session_id,
                            }
                        )
            return out

        def _try_subject_layout():
            out = []
            subject_dirs = sorted(glob.glob(os.path.join(data_path, "subject*")))
            for subject_dir in subject_dirs:
                if not os.path.isdir(subject_dir):
                    continue
                subject_id = os.path.basename(subject_dir)

                rgb_video = os.path.join(subject_dir, "Output_RGB.mkv")
                mindray_file = os.path.join(subject_dir, "recorded_mindray.txt")

                if os.path.exists(rgb_video) and os.path.exists(mindray_file):
                    m = re.search(r"(\d+)", subject_id)
                    subject = int(m.group(1)) if m else -1
                    out.append(
                        {
                            "index": subject_id,
                            "path": subject_dir,
                            "subject": subject,
                            "patient_id": subject_id,
                            "session_id": "0",
                        }
                    )
            return out

        # First try the original patient/session layout
        data_dirs = _try_patient_session_layout()
        # Fallback to subjectXX layout
        if not data_dirs:
            data_dirs = _try_subject_layout()

        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits."""
        
        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:
                data_info[subject] = []
            data_info[subject].append(data)

        subj_list = list(data_info.keys())
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi_process."""
        saved_filename = data_dirs[i]['index']
        session_path = data_dirs[i]['path']

        # Read Frames
        video_type = getattr(config_preprocess, 'VIDEO_TYPE', 'RGB')
        if video_type == 'RGB':
            video_file = os.path.join(session_path, "Output_RGB.mkv")
        elif video_type == 'IR1':
            video_file = os.path.join(session_path, "Output_IR1.mkv")
        elif video_type == 'IR2':
            video_file = os.path.join(session_path, "Output_IR2.mkv")
        else:
            video_file = os.path.join(session_path, "Output_RGB.mkv")

        if 'None' in config_preprocess.DATA_AUG:
            use_streaming = (
                not getattr(config_preprocess, 'USE_PSUEDO_PPG_LABEL', False)
                and getattr(config_preprocess, 'DO_CHUNK', False)
                and 'Raw' in config_preprocess.DATA_TYPE
                and len(config_preprocess.DATA_TYPE) == 1
            )
            if use_streaming:
                input_name_list = self.preprocess_video_streaming(
                    video_file, session_path, config_preprocess, saved_filename
                )
                file_list_dict[i] = input_name_list
                return
            frames = self.read_video(video_file)
        elif 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(
                glob.glob(os.path.join(session_path, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels (PPG from Mindray)
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            mindray_file = os.path.join(session_path, "recorded_mindray.txt")
            timestamp_file = os.path.join(session_path, "timestamp_log.txt")
            bvps = self.read_mindray_ppg(mindray_file, timestamp_file)

        # Resample PPG to match video frame count
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def preprocess_video_streaming(self, video_file, session_path, config_preprocess, saved_filename):
        """Stream video frames and save preprocessed chunks to avoid full-video memory load."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)

        frame_count = self.get_video_frame_count(video_file)
        if frame_count <= 0:
            raise ValueError(f"Invalid frame count from video: {video_file}")

        mindray_file = os.path.join(session_path, "recorded_mindray.txt")
        timestamp_file = os.path.join(session_path, "timestamp_log.txt")
        bvps = self.read_mindray_ppg(mindray_file, timestamp_file)
        bvps = BaseLoader.resample_ppg(bvps, frame_count)

        crop_cfg = config_preprocess.CROP_FACE
        face_boxes = self._prepare_stream_face_boxes(video_file, crop_cfg)

        chunk_length = config_preprocess.CHUNK_LENGTH
        input_path_name_list = []
        frame_buffer = []
        chunk_id = 0
        frame_idx = 0

        cap = cv2.VideoCapture(video_file)
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            processed_frame = self._crop_resize_stream_frame(frame, frame_idx, face_boxes, crop_cfg, config_preprocess.RESIZE.W, config_preprocess.RESIZE.H)
            frame_buffer.append(processed_frame)
            frame_idx += 1

            if len(frame_buffer) == chunk_length:
                start_idx = frame_idx - chunk_length
                end_idx = frame_idx
                frames_chunk = np.asarray(frame_buffer)
                label_chunk = bvps[start_idx:end_idx]
                frames_chunk, label_chunk = self._apply_stream_transforms(frames_chunk, label_chunk, config_preprocess)

                input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(saved_filename, str(chunk_id))
                label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(saved_filename, str(chunk_id))
                np.save(input_path_name, frames_chunk)
                np.save(label_path_name, label_chunk)
                input_path_name_list.append(input_path_name)

                chunk_id += 1
                frame_buffer = []

        cap.release()

        if not input_path_name_list:
            raise ValueError(f"No chunks generated for {saved_filename}. frame_count={frame_count}, chunk_length={chunk_length}")

        return input_path_name_list

    def _prepare_stream_face_boxes(self, video_file, crop_cfg):
        """Prepare face boxes for streaming crop path."""
        if not crop_cfg.DO_CROP_FACE:
            return {'mode': 'full'}

        if not crop_cfg.DETECTION.DO_DYNAMIC_DETECTION:
            cap = cv2.VideoCapture(video_file)
            success, frame = cap.read()
            cap.release()
            if not success:
                raise ValueError(f"Cannot read first frame from video: {video_file}")
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            box = self.face_detection(frame, crop_cfg.BACKEND, crop_cfg.USE_LARGE_FACE_BOX, crop_cfg.LARGE_BOX_COEF)
            return {'mode': 'single', 'box': np.asarray(box, dtype='int')}

        detection_freq = max(1, int(crop_cfg.DETECTION.DYNAMIC_DETECTION_FREQUENCY))
        boxes = []

        cap = cv2.VideoCapture(video_file)
        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_idx % detection_freq == 0:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                box = self.face_detection(frame, crop_cfg.BACKEND, crop_cfg.USE_LARGE_FACE_BOX, crop_cfg.LARGE_BOX_COEF)
                boxes.append(np.asarray(box, dtype='int'))
            frame_idx += 1
        cap.release()

        if not boxes:
            raise ValueError(f"Dynamic face detection produced no boxes for video: {video_file}")

        if crop_cfg.DETECTION.USE_MEDIAN_FACE_BOX:
            return {'mode': 'median', 'box': np.median(np.asarray(boxes), axis=0).astype('int')}

        return {'mode': 'dynamic', 'boxes': boxes, 'freq': detection_freq}

    @staticmethod
    def _crop_resize_stream_frame(frame, frame_idx, face_boxes, crop_cfg, width, height):
        """Crop and resize one frame using prepared stream face boxes."""
        mode = face_boxes['mode']
        if mode == 'full':
            region = [0, 0, frame.shape[1], frame.shape[0]]
        elif mode == 'single':
            region = face_boxes['box']
        elif mode == 'median':
            region = face_boxes['box']
        elif mode == 'dynamic':
            ref_idx = frame_idx // face_boxes['freq']
            if ref_idx >= len(face_boxes['boxes']):
                ref_idx = len(face_boxes['boxes']) - 1
            region = face_boxes['boxes'][ref_idx]
        else:
            region = [0, 0, frame.shape[1], frame.shape[0]]

        if crop_cfg.DO_CROP_FACE:
            x, y, w, h = [int(v) for v in region]
            frame = frame[max(y, 0):min(y + h, frame.shape[0]), max(x, 0):min(x + w, frame.shape[1])]

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _apply_stream_transforms(frames_chunk, label_chunk, config_preprocess):
        """Apply frame/label transforms for stream preprocessing mode."""
        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            if data_type == "Raw":
                data.append(frames_chunk.copy())
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(frames_chunk.copy()))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(frames_chunk.copy()))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)

        if config_preprocess.LABEL_TYPE == "Raw":
            processed_label = label_chunk
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            processed_label = BaseLoader.diff_normalize_label(label_chunk)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            processed_label = BaseLoader.standardized_label(label_chunk)
        else:
            raise ValueError("Unsupported label type!")

        return data, processed_label

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3)"""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    @staticmethod
    def read_mindray_ppg(mindray_file, timestamp_file=None):
        """Reads PPG signal from Mindray HL7 data file.
        
        Args:
            mindray_file(str): path to recorded_mindray.txt
            timestamp_file(str): path to timestamp_log.txt (optional)
            
        Returns:
            ppg_signal(np.array): concatenated PPG waveform values
        """
        ppg_values = []
        
        with open(mindray_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse HL7 messages for PPG data
        # PPG data is in OBX segments with MDC_PULS_OXIM_PLETH
        # Format: OBX|1|NA|150452^MDC_PULS_OXIM_PLETH^MDC|...|values^values^...|...
        
        lines = content.split('\n')
        for line in lines:
            if 'MDC_PULS_OXIM_PLETH' in line and line.startswith('OBX'):
                # Parse HL7 OBX segment
                fields = line.split('|')
                if len(fields) >= 6:
                    # Field 5 (index 5) contains the waveform values separated by ^
                    value_field = fields[5]
                    if value_field:
                        values = value_field.split('^')
                        for v in values:
                            try:
                                ppg_values.append(float(v))
                            except ValueError:
                                continue
        
        if not ppg_values:
            raise ValueError(f"No PPG data found in {mindray_file}")
        
        return np.asarray(ppg_values)

    @staticmethod
    def read_mindray_ecg(mindray_file, lead='II'):
        """Reads ECG signal from Mindray HL7 data file.
        
        Args:
            mindray_file(str): path to recorded_mindray.txt
            lead(str): ECG lead to extract ('I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V')
            
        Returns:
            ecg_signal(np.array): concatenated ECG waveform values
        """
        ecg_values = []
        
        # Map lead names to MDC codes
        lead_map = {
            'I': 'MDC_ECG_ELEC_POTL_I',
            'II': 'MDC_ECG_ELEC_POTL_II',
            'III': 'MDC_ECG_ELEC_POTL_III',
            'AVR': 'MDC_ECG_ELEC_POTL_AVR',
            'AVL': 'MDC_ECG_ELEC_POTL_AVL',
            'AVF': 'MDC_ECG_ELEC_POTL_AVF',
            'V': 'MDC_ECG_ELEC_POTL_V'
        }
        
        mdc_code = lead_map.get(lead.upper(), 'MDC_ECG_ELEC_POTL_II')
        
        with open(mindray_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        for line in lines:
            if mdc_code in line and line.startswith('OBX'):
                fields = line.split('|')
                if len(fields) >= 6:
                    value_field = fields[5]
                    if value_field:
                        values = value_field.split('^')
                        for v in values:
                            try:
                                ecg_values.append(float(v))
                            except ValueError:
                                continue
        
        if not ecg_values:
            raise ValueError(f"No ECG data (lead {lead}) found in {mindray_file}")
        
        return np.asarray(ecg_values)

    @staticmethod
    def read_mindray_resp(mindray_file):
        """Reads respiratory signal from Mindray HL7 data file.
        
        Args:
            mindray_file(str): path to recorded_mindray.txt
            
        Returns:
            resp_signal(np.array): concatenated respiratory waveform values
        """
        resp_values = []
        
        with open(mindray_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Respiratory data is in MDC_IMPED_TTHOR
        lines = content.split('\n')
        for line in lines:
            if 'MDC_IMPED_TTHOR' in line and line.startswith('OBX'):
                fields = line.split('|')
                if len(fields) >= 6:
                    value_field = fields[5]
                    if value_field:
                        values = value_field.split('^')
                        for v in values:
                            try:
                                resp_values.append(float(v))
                            except ValueError:
                                continue
        
        if not resp_values:
            raise ValueError(f"No respiratory data found in {mindray_file}")
        
        return np.asarray(resp_values)

    @staticmethod
    def parse_timestamp_log(timestamp_file):
        """Parse timestamp log file.
        
        Args:
            timestamp_file(str): path to timestamp_log.txt
            
        Returns:
            dict: dictionary containing timestamp information
        """
        timestamps = {}
        
        with open(timestamp_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if 'Camera open time' in line:
                # Parse: Camera open time (system): 2025-09-16 17:13:49.842
                match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line)
                if match:
                    timestamps['camera_open'] = match.group(1)
            elif 'Mindray first update time' in line:
                match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if match:
                    timestamps['mindray_first_update'] = match.group(1)
            elif 'Mindray log trigger time' in line:
                match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line)
                if match:
                    timestamps['mindray_trigger'] = match.group(1)
        
        # Parse frame offset if present (last line with just a number)
        for line in reversed(lines):
            line = line.strip()
            if line.isdigit():
                timestamps['frame_offset'] = int(line)
                break
        
        return timestamps

    @staticmethod
    def get_video_fps(video_file):
        """Get the FPS of a video file.
        
        Args:
            video_file(str): path to video file
            
        Returns:
            float: frames per second
        """
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    @staticmethod
    def get_video_frame_count(video_file):
        """Get the total frame count of a video file.
        
        Args:
            video_file(str): path to video file
            
        Returns:
            int: total number of frames
        """
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

