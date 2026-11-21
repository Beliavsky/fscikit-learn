! Native calibration utilities; algorithm implementation/provenance in sklearn_ranking.
module sklearn_calibration
  use sklearn_calibrated_classifier, only: calibrated_classifier_cv
  use sklearn_ranking, only: calibration_curve
  implicit none
  private
  public :: calibration_curve
  public :: calibrated_classifier_cv
end module
