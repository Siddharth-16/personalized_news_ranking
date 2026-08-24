# Data Split and Leakage Policy

## Dataset

Development uses Microsoft MINDsmall.

The provided `MINDsmall_train` behavior data spans November 9–14,
2019 and contains 156,965 impressions.

## Development Split

`MINDsmall_train` is divided chronologically:

- Training: November 9–13, 2019

  - 126,695 impressions
  - 80.72% of the data

- Internal validation: November 14, 2019
  - 30,270 impressions
  - 19.28% of the data

The cutoff is November 14, 2019 at 00:00:00.

The split satisfies:

`max(train.time) < min(validation.time)`

This prevents future impressions from being included in the
training period.

## Final Holdout

`MINDsmall_dev` is kept untouched during model development and
will be used as the final holdout dataset.

## Leakage Policy

Statistics learned from interactions, such as article popularity,
are computed only from training-period impressions.

Validation impression labels are never used to construct features
or ranking scores.

The click history supplied with a MIND impression may be used for
personalization because it represents user interactions preceding
that recommendation event.

Article metadata such as category and title may be used for both
training and validation articles because it is information available
about the article itself rather than future interaction outcomes.
