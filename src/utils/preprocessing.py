import os
import gc
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from contextlib import redirect_stdout


class PANDAImagePreprocessing:
    """
    画像の読み込み＆前処理＆保存
    指定した画像サイズに応じてパディングを行い分割する
    分割した中のマスク（gleason_score）の割合を計算する

    """
    
    def __init__(self, target_id, img_size=128, background_rate=0.5, data_dir='../data/input', save_dir='.'):
        self.id = target_id
        self.img_size = img_size
        self.background_rate = background_rate
        self.data_dir = data_dir
        self.save_dir = save_dir

    def _display_img(self):
        """
        正常な画像を読み込む
        :return: ndarray:
        """
        img_path = os.path.join(self.data_dir, 'train_images', f'{self.id}.tiff')
        # Using Openslide
        slide = openslide.OpenSlide(img_path)
        # Set Properties  1: Point   2: Tiff Level   3: Viewing Dimension
        # .level_count -> Get Tiff Level Count
        # .level_dimensions -> Get Tiff Width, Height per Level
        try:
            patch = slide.read_region((0, 0), 0, slide.level_dimensions[0])
        except:
            return None

        # PIL -> ndarray
        patch = np.asarray(patch)
        # RGBA -> RGB
        if patch.shape[-1] == 4:
            patch = patch[:, :, :3]

        slide.close()

        return patch

    def _display_mask(self, data_provider='radboud'):
        """
        マスクデータを読み込む
        :param data_provider: 
        :return: ndarray
        """
        assert data_provider in ['radboud', 'karolinska'], "Please Set center=['radboud', 'karolinska']"

        img_path = os.path.join(self.data_dir, 'train_label_masks', f'{self.id}_mask.tiff')
        # Using Openslide
        slide = openslide.OpenSlide(img_path)
        # Set Properties  1: Point   2: Tiff Level   3: Viewing Dimension
        # .level_count -> Get Tiff Level Count
        # .level_dimensions -> Get Tiff Width, Height per Level
        # なぜか読み込めないものもある　一旦例外処理する
        try:
            mask_data = slide.read_region((0, 0), 0, slide.level_dimensions[0])
        except:
            return None

        mask_data = mask_data.split()[0]
        # To show the masks we map the raw label values to RGB values
        preview_palette = np.zeros(shape=768, dtype=int)
        if data_provider == 'radboud':
            # Mapping
            preview_palette[0:18] = (np.array(
                [0, 0, 0,         # background(黒)
                 51, 51, 51,      # stroma(濃灰色)
                 102, 102, 102,   # benign epithelium(淡灰色)
                 255, 255, 178,   # Gleason 3(淡黄色)
                 255, 127, 0,     # Gleason 4(オレンジ)
                 255, 0, 0]       # Gleason 5(赤)
            )).astype(int)
            
        elif data_provider == 'karolinska':
            # Mapping
            preview_palette[0:9] = (np.array(
                [0, 0, 0,         # background(黒)
                 127, 127, 127,   # benign(灰色)
                 255, 0, 0]       # cancer(赤色)
            )).astype(int)
            
        mask_data.putpalette(data=preview_palette.tolist())
        mask_data = mask_data.convert(mode='RGB')

        # PIL -> ndarray
        mask_data = np.asarray(mask_data)
        # RGBA -> RGB
        if mask_data.shape[-1] == 4:
            mask_data = mask_data[:, :, :3]

        slide.close()

        return mask_data

    def transform(self):
        """
        前処理を実行
        画像とマスクデータを読み込む
        img_size×img_sizeのグリッドに合うようにpaddingを行い、
        グリッドごとのマスクデータからそのグリッドにおけるgleason_scoreを集計する
        グリッドごとの画像をjpg形式で保存（背景の割合がbackground_rate以上のものは無視する）
        :return: dataframe グリッドごとのgleason_scoreの集計結果
        """
        print('Start Preprocessing')
        print('Target Image ID: ', self.id)
        res = pd.DataFrame()

        print('Image Loading...')
        img = self._display_img()
        mask = self._display_mask()
        if img is None or mask is None:
            return None

        # Padding
        H, W = img.shape[:2]
        pad_h = (self.img_size - H % self.img_size) % self.img_size
        pad_w = (self.img_size - W % self.img_size) % self.img_size

        padded_img = np.pad(
            img,
            pad_width=[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
            constant_values=255
        )

        padded_mask = np.pad(
            mask,
            pad_width=[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
            constant_values=0
        )

        new_H, new_W = padded_img.shape[:2]

        del img, mask
        gc.collect()

        print('Calculating Score per Grid')
        for h in range(int(new_H / self.img_size)):
            for w in range(int(new_W / self.img_size)):
                # トリミングする
                _mask = padded_mask[h * self.img_size:(h + 1) * self.img_size,
                                    w * self.img_size:(w + 1) * self.img_size, :]
                # Channelで足す
                _mask = np.sum(_mask, axis=2)

                # 0: 背景
                # 1: stroma(灰色)  = 153
                _mask = np.where(_mask == 153, 0, _mask)
                # 2: benign epithelium(緑)  = 306
                _mask = np.where(_mask == 306, 0, _mask)
                # 3: Gleason 3(淡黄色)  255, 255, 178 = = 688  score=3
                _mask = np.where(_mask == 688, 3, _mask)
                # 4: Gleason 4(オレンジ)   255, 127, 0 = 382   score=4
                _mask = np.where(_mask == 382, 4, _mask)
                # 5: Gleason 5(赤)   255, 0, 0 = (255 * 1.0 = 255)   score=5
                _mask = np.where(_mask == 255, 5, _mask)

                u, counts = np.unique(_mask, return_counts=True)
                _dict = {f'score_{k}': [v] for k, v in zip(u, counts)}

                for score in ['score_0', 'score_3', 'score_4', 'score_5']:
                    if score not in _dict:
                        _dict[score] = [0]

                _dict['image_id'] = [self.id + f'_{h}_{w}']  # グリッドのインデックスをファイル名に付与
                _res = pd.DataFrame(_dict)

                res = pd.concat([res, _res], axis=0, ignore_index=True)

        # 割合を計算（1グリッドごとの全ピクセルで除算）
        res['score_0'] = res['score_0'] / (self.img_size * self.img_size)
        res['score_3'] = res['score_3'] / (self.img_size * self.img_size)
        res['score_4'] = res['score_4'] / (self.img_size * self.img_size)
        res['score_5'] = res['score_5'] / (self.img_size * self.img_size)

        # 背景が多い画像は対象外とする
        _res = res[res['score_0'] < self.background_rate].reset_index(drop=True)
        # 対象の画像をトリミングし保存する
        print('Saving Image Num: ', len(_res))
        for i in range(len(_res)):
            image_id = _res['image_id'].loc[i]
            save_img_h = int(image_id.split('_')[1])
            save_img_w = int(image_id.split('_')[2])
            # 画像をトリミング
            _img = padded_img[save_img_h * self.img_size:(save_img_h + 1) * self.img_size,
                              save_img_w * self.img_size:(save_img_w + 1) * self.img_size, :]
            _mask = padded_mask[save_img_h * self.img_size:(save_img_h + 1) * self.img_size,
                                save_img_w * self.img_size:(save_img_w + 1) * self.img_size, :]

            # PIL形式に変換しjpgで保存
            _img = Image.fromarray(_img)
            _img.save(os.path.join(self.save_dir, image_id + '.jpg'))

        print('Finish')
        print('#'*30)

        return _res
