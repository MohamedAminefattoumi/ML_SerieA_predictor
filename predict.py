import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import difflib
import re


def _parse_round_index(round_value):
	if pd.api.types.is_number(round_value):
		return int(round_value)
	if isinstance(round_value, str):
		parts = [int(s) for s in round_value.split() if s.isdigit()]
		return parts[0] if parts else 0
	return 0


def _build_training_artifacts(dataset_path="Serie_A.csv"):
	# Load raw dataset
	dataset = pd.read_csv(dataset_path)

	# Compute pre-match averages exactly like training
	team_avg_GF = []
	team_avg_GA = []
	team_avg_xG = []
	team_avg_xGA = []
	team_avg_Poss = []
	opp_avg_GF = []
	opp_avg_GA = []
	opp_avg_xG = []
	opp_avg_xGA = []
	opp_avg_Poss = []

	# Helper index to compare previous rounds robustly
	dataset["_RoundIndex"] = dataset["Round"].apply(_parse_round_index)

	for _, row in dataset.iterrows():
		season = row["Season"]
		team = row["Team"]
		opponent = row["Opponent"]
		round_idx = row["_RoundIndex"]

		team_history = dataset[(dataset["Season"] == season) & (dataset["Team"] == team) & (dataset["_RoundIndex"] < round_idx)]
		opp_history = dataset[(dataset["Season"] == season) & (dataset["Team"] == opponent) & (dataset["_RoundIndex"] < round_idx)]

		team_avg_GF.append(team_history["GF"].mean() if not team_history.empty else np.nan)
		team_avg_GA.append(team_history["GA"].mean() if not team_history.empty else np.nan)
		team_avg_xG.append(team_history["xG"].mean() if not team_history.empty else np.nan)
		team_avg_xGA.append(team_history["xGA"].mean() if not team_history.empty else np.nan)
		team_avg_Poss.append(team_history["Poss"].mean() if not team_history.empty else np.nan)

		opp_avg_GF.append(opp_history["GF"].mean() if not opp_history.empty else np.nan)
		opp_avg_GA.append(opp_history["GA"].mean() if not opp_history.empty else np.nan)
		opp_avg_xG.append(opp_history["xG"].mean() if not opp_history.empty else np.nan)
		opp_avg_xGA.append(opp_history["xGA"].mean() if not opp_history.empty else np.nan)
		opp_avg_Poss.append(opp_history["Poss"].mean() if not opp_history.empty else np.nan)

	dataset["Team_GF_avg"] = team_avg_GF
	dataset["Team_GA_avg"] = team_avg_GA
	dataset["Team_xG_avg"] = team_avg_xG
	dataset["Team_xGA_avg"] = team_avg_xGA
	dataset["Team_Poss_avg"] = team_avg_Poss
	dataset["Opponent_GF_avg"] = opp_avg_GF
	dataset["Opponent_GA_avg"] = opp_avg_GA
	dataset["Opponent_xG_avg"] = opp_avg_xG
	dataset["Opponent_xGA_avg"] = opp_avg_xGA
	dataset["Opponent_Poss_avg"] = opp_avg_Poss

	pre_match_features = [
		"Season",
		"Round",
		"Team",
		"Opponent",
		"Venue",
		"Team_GF_avg",
		"Team_GA_avg",
		"Team_Poss_avg",
		"Opponent_GF_avg",
		"Opponent_GA_avg",
		"Opponent_Poss_avg",
		"Result",
	]
	data_pre = dataset[pre_match_features].copy()

	# Imputer for numerical averages
	imputer = SimpleImputer(strategy="mean")
	num_cols = [
		"Team_GF_avg",
		"Team_GA_avg",
		"Team_Poss_avg",
		"Opponent_GF_avg",
		"Opponent_GA_avg",
		"Opponent_Poss_avg",
	]
	data_pre.loc[:, num_cols] = imputer.fit_transform(data_pre[num_cols])

	# Encode Venue exactly like training (LabelEncoder)
	le_venue = LabelEncoder()
	data_pre.loc[:, "Venue"] = le_venue.fit_transform(data_pre["Venue"])

	# OneHot for categorical features (same settings)
	categorical_features = ["Season", "Round", "Team", "Opponent"]
	ct = ColumnTransformer(
		transformers=[("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)],
		remainder="passthrough",
	)
	X_like = data_pre.drop("Result", axis=1)
	ct.fit(X_like)

	# Label encoder for target to decode predictions later
	le_result = LabelEncoder().fit(data_pre["Result"])

	return {
		"dataset": dataset,
		"data_pre": data_pre,
		"imputer": imputer,
		"le_venue": le_venue,
		"ct": ct,
		"le_result": le_result,
	}


def _normalize_season(season_input, season_series):
	# Convert to integer when possible to match dataset dtype
	try:
		season_int = int(str(season_input).strip())
		return season_int
	except Exception:
		# Fallback: return as-is
		return season_input


def _normalize_venue(venue_input):
	text = str(venue_input).strip().lower()
	if text in {"h", "home", "host"}:
		return "Home"
	if text in {"a", "away", "visitor", "visitors"}:
		return "Away"
	# Title case common values
	return text.title()


def _normalize_round(round_input):
	# Accept integers or strings like "Matchweek 12"
	if pd.api.types.is_number(round_input):
		return f"Matchweek {int(round_input)}"
	text = str(round_input).strip()
	nums = re.findall(r"\d+", text)
	if nums:
		return f"Matchweek {int(nums[0])}"
	return text


def _normalize_team(name_input, known_teams):
	# First try exact and case-insensitive match
	name = str(name_input).strip()
	if name in known_teams:
		return name
	lookup = {t.lower(): t for t in known_teams}
	low = name.lower()
	if low in lookup:
		return lookup[low]
	# Fuzzy match on lowercase names
	candidates = list(lookup.keys())
	match = difflib.get_close_matches(low, candidates, n=1, cutoff=0.6)
	if match:
		return lookup[match[0]]
	# As a last resort, return the original (will likely fail validation later)
	return name


def predict_match_result(team, opponent, venue, season, round_value, model_path="best_model.pkl"):
	"""
	Predict W/D/L for a given Serie A fixture.

	Parameters:
	- team (str)
	- opponent (str)
	- venue (str): "Home" or "Away"
	- season (int or str): must exist in training data
	- round_value (int or str): e.g. 12 or "Matchweek 12"
	- model_path (str): path to the saved model

	Returns: (label, probabilities_dict or None)
	"""
	artifacts = _build_training_artifacts()
	dataset = artifacts["dataset"]
	imputer = artifacts["imputer"]
	le_venue = artifacts["le_venue"]
	ct = artifacts["ct"]
	le_result = artifacts["le_result"]

	# Normalize inputs to align with training data
	known_teams = sorted(dataset["Team"].dropna().unique())
	team = _normalize_team(team, known_teams)
	opponent = _normalize_team(opponent, known_teams)
	venue = _normalize_venue(venue)
	season = _normalize_season(season, dataset["Season"])  # cast to int when possible
	round_value = _normalize_round(round_value)

	# Validate after normalization
	if team not in dataset["Team"].unique():
		raise ValueError(f"Unknown team: {team}")
	if opponent not in dataset["Team"].unique():
		raise ValueError(f"Unknown opponent: {opponent}")
	if season not in dataset["Season"].unique():
		raise ValueError(f"Unknown season: {season}")
	if venue not in le_venue.classes_:
		raise ValueError(f"Unknown venue: {venue}. Expected one of {list(le_venue.classes_)}")

	# Compute pre-match averages from history up to the provided round
	dataset["_RoundIndex"] = dataset["Round"].apply(_parse_round_index)
	round_idx = _parse_round_index(round_value)

	team_history = dataset[(dataset["Season"] == season) & (dataset["Team"] == team) & (dataset["_RoundIndex"] < round_idx)]
	opp_history = dataset[(dataset["Season"] == season) & (dataset["Team"] == opponent) & (dataset["_RoundIndex"] < round_idx)]

	row = {
		"Season": season,
		"Round": round_value,
		"Team": team,
		"Opponent": opponent,
		"Venue": venue,
		"Team_GF_avg": team_history["GF"].mean() if not team_history.empty else np.nan,
		"Team_GA_avg": team_history["GA"].mean() if not team_history.empty else np.nan,
		"Team_Poss_avg": team_history["Poss"].mean() if not team_history.empty else np.nan,
		"Opponent_GF_avg": opp_history["GF"].mean() if not opp_history.empty else np.nan,
		"Opponent_GA_avg": opp_history["GA"].mean() if not opp_history.empty else np.nan,
		"Opponent_Poss_avg": opp_history["Poss"].mean() if not opp_history.empty else np.nan,
	}
	row_df = pd.DataFrame([row])

	# Impute missing averages with training means
	num_cols = [
		"Team_GF_avg",
		"Team_GA_avg",
		"Team_Poss_avg",
		"Opponent_GF_avg",
		"Opponent_GA_avg",
		"Opponent_Poss_avg",
	]
	row_df.loc[:, num_cols] = imputer.transform(row_df[num_cols])

	# Encode venue with the same LabelEncoder
	row_df.loc[:, "Venue"] = le_venue.transform(row_df["Venue"])

	# Ensure column order matches training X
	X_columns = [
		"Season",
		"Round",
		"Team",
		"Opponent",
		"Venue",
		"Team_GF_avg",
		"Team_GA_avg",
		"Team_Poss_avg",
		"Opponent_GF_avg",
		"Opponent_GA_avg",
		"Opponent_Poss_avg",
	]
	row_df = row_df[X_columns]

	# Transform with the fitted ColumnTransformer
	X_row = ct.transform(row_df)

	# Load model and predict
	model = joblib.load(model_path)
	pred_int = model.predict(X_row)[0]
	pred_label = le_result.inverse_transform([pred_int])[0]

	probas = None
	if hasattr(model, "predict_proba"):
		try:
			p = model.predict_proba(X_row)[0]
			classes = le_result.classes_
			probas = {cls: float(prob) for cls, prob in zip(classes, p)}
		except Exception:
			probas = None

	return pred_label, probas


if __name__ == "__main__":
	try:
		team = input("Enter the team: ")
		opponent = input("Enter the opponent: ")
		venue = input("Enter the venue (Home/Away): ")
		season = input("Enter the season (e.g., 2025): ")
		round_value = input("Enter the round value (e.g., Matchweek 10 or 10): ")

		label, probas = predict_match_result(
			team,
			opponent,
			venue,
			season,
			round_value,
		)
		print("Prediction:", label)
	
	except Exception as exc:
		print("Error:", exc)



