# preprocessor.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import pandas as pd
import os

# all datasets need:
# 1. source
# 2. destination
# 3. distance
# 4. source lat
# 5. source lng
# 6. destination lat
# 7. destination lng


class preprocessor:

    @staticmethod
    def _save(
        df: pd.DataFrame,
        targetType: str = "parquet"
    ) -> None:
        """
        Save the DataFrame to a file in the data directory.

        Parameters:
            df (pd.DataFrame): The DataFrame to be saved.
            targetType (str): The type of file to be saved. Defaults to "parquet".
        """
        from pathlib import Path

        # create data directory if it doesn't exist (should be on te same level as this parent folder)
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / f"fullDataset.{targetType}"

        if targetType == "csv":
            # Save the DataFrame to a csv file
            df.to_csv(file_path, index=False)
        else:
            # Save the DataFrame to a parquet file
            df.to_parquet(file_path, engine="pyarrow")

    @staticmethod
    def preprocess(
        path: str,
        sourceKey: str = "source",
        sourceNameKey: str = "source_name",
        destinationKey: str = "destination",
        destinationNameKey: str = "destination_name",
        distanceKey: str = "distance",
        sourceLatKey: str = "source_lat",
        sourceLngKey: str = "source_lng",
        destinationLatKey: str = "destination_lat",
        destinationLngKey: str = "destination_lng",
        targetType: str = "parquet"
    ) -> pd.DataFrame:
        """
        Preprocess a dataset by renaming columns to the desired format,
        calculating distances and adding the result to the dataframe.

        Parameters:
        path (str): path to the dataset
        sourceKey (str): key for the source column (default: "source")
        destinationKey (str): key for the destination column (default: "destination")
        distanceKey (str): key for the distance column (default: "distance")
        sourceLatKey (str): key for the source latitude column (default: "source_lat")
        sourceLngKey (str): key for the source longitude column (default: "source_lng")
        destinationLatKey (str): key for the destination latitude column (default: "destination_lat")
        destinationLngKey (str): key for the destination longitude column (default: "destination_lng")

        Returns:
        pd.DataFrame: the preprocessed dataframe
        """

        # check if file exists and read it into a df
        _, fType = os.path.splitext(path)
        if fType == ".csv":
            df = pd.read_csv(path)
        elif fType == ".parquet":
            df = pd.read_parquet(path)

        # get all column names
        cols = list(df.columns)

        # check if all required columns are present
        if any([
            sourceKey not in cols,
            sourceNameKey not in cols,
            destinationKey not in cols,
            destinationNameKey not in cols,
            sourceLatKey not in cols,
            sourceLngKey not in cols,
            destinationLatKey not in cols,
            destinationLngKey not in cols
        ]):
            raise Exception("Invalid dataset")

        # rename columns to the desired format
        df.rename(columns={
            sourceKey: "source",
            sourceNameKey: "source_name",
            destinationKey: "destination",
            destinationNameKey: "destination_name",
            sourceLatKey: "source_lat",
            sourceLngKey: "source_lng",
            destinationLatKey: "destination_lat",
            destinationLngKey: "destination_lng",
            **({distanceKey: "distance"} if distanceKey in cols else {})
        }, inplace=True)

        # distance is already present return here
        if distanceKey in cols:
            preprocessor._save(df, targetType=targetType)
            return df[[
                "source",
                "source_name",
                "destination",
                "destination_name",
                "distance",
                "source_lat",
                "source_lng",
                "destination_lat",
                "destination_lng"
            ]]

        # calculate distance
        df["distance"] = preprocessor.haversine(df)

        # save df
        preprocessor._save(df, targetType=targetType)
        # return processed df
        return df[[
            "source",
            "source_name"
            "destination",
            "destination_name",
            "distance",
            "source_lat",
            "source_lng",
            "destination_lat",
            "destination_lng"
        ]]

    @staticmethod
    def haversine(df: pd.DataFrame) -> float:
        # use torch for vector calculation
        import torch
        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # compute vectorized haversine
        with torch.no_grad():
            # convert to radians
            lat1 = torch.deg2rad(torch.tensor(df["source_lat"].values, device=device))
            lng1 = torch.deg2rad(torch.tensor(df["source_lng"].values, device=device))
            lat2 = torch.deg2rad(torch.tensor(df["destination_lat"].values, device=device))
            lng2 = torch.deg2rad(torch.tensor(df["destination_lng"].values, device=device))

            # compute delta lat and delta lng
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            # compute haversine
            a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2)**2
            c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

            distances = 6371 * c

            return distances.cpu().numpy()

    @staticmethod
    def combine(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        # Combine the two DataFrames
        combined_df = pd.concat([df1, df2], axis=0)
        return combined_df
