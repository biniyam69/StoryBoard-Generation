import cv2
import numpy as np
import os, sys,glob
from pathlib import Path
from typing import Union
from match import ImageMatcher
import pandas as pd


class Extractor:
    
    """
    A simple class for feature extraction from image assets.

    Attribs:
        data_folder (Union[str, Path]): The full path to the assets folder.
        extracted_path (Path): The directory where extracted features will be saved.
    """
    
    def __init__(self, data_folder: Union[str, Path]) -> None:
        self.assets_folder = Path(data_folder)
        self.extracted_path = self.assets_folder.parent / "extracted_features"
        self.extracted_path.mkdir(parents=True, exist_ok=True)
        
        def _extract_features(self, feature_name: str, template_name: str) -> pd.DataFrame:
            """
            Extracts the location of a specific feature from the preview images in the asset
            
            Args:
                feature_name (str): The name of the feature to extract
                template_name (str): The name of the template image to use for matching
            
            Returns:
                pd.DataFrame: A dataframe containing the location of the feature in the image
                
            """
            t_matching = ImageMatcher('img')
            feature_positions = []
            
            for folder in self.assets_folder.glob("*"):
                query_img = folder / "_preview.png"
                
                train_img = folder / f"{template_name}.png"
                
                if query_img.exists() and train_img.exists():
                    location, bottom_right, top_left, _, _ = t_matching.template_matching(
                        str(train_img), str(query_img), method = cv2.TM_CCOEFF_NORMED
                    )
                    
                    if all((location, bottom_right, top_left)):
                        feature_positions.append([folder.stem, *location, *bottom_right, *top_left])
                        
                    else:
                        feature_positions.append([folder.stem] + [0] * 6)
                else:
                    feature_positions.append([folder.stem] + [0] * 6)
            columns = ['id', f'{feature_name}_x', f'{feature_name}_y',
                       f'{feature_name}_br_x', f'{feature_name}_br_y',
                       f'{feature_name}_tl_x', f'{feature_name}_tl_y']
            
            return pd.DataFrame(feature_positions, columns=columns)
        
        def segment_etractor(self, segment_name: str) -> pd.DataFrame:
            """
            Extracts the location of a specific feature from the preview images in the asset
            
            Args:
                segment_name (str): The name of the segment to extract
            
            Returns:
                pd.DataFrame: A dataframe containing the location of the feature in the image
                
            """
            return self._extract_features(segment_name, segment_name)
        
        def logo(self) -> pd.DataFrame:
            """
            Extracts the location of the logo from the preview images in the asset
            
            Returns:
            pd.DataFrame: A dataframe containing the location of the feature in the image
            """
            return self._extract_features("logo", "logo")
        
        def engagement_button(self) -> pd.DataFrame:
            """
            Extracts the location of the engagement button from the preview images in the asset
            
            Returns:
            pd.DataFrame: A dataframe containing the location of the feature in the image
            """
            return self._extract_features("engagement", "engagement")
        
        def extract_cta(self) -> pd.DataFrame:
            """
            Extracts the location of the call to action from the preview images in the asset
            
            Returns:
            pd.DataFrame: A dataframe containing the location of the feature in the image
            """
            return self._extract_features("cta", "cta")
        

if __name__ == "__main__":
    data_folder = "/data/assets"
    extractor = Extractor(data_folder)
    segment_df = extractor.segment_extractor("segment")
    logo_df = extractor.logo()
    engagement_df = extractor.engagement_button()
    cta_df = extractor.extract_cta()
    