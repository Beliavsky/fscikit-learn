program test_univariate_selection
  use sklearn_univariate_selection
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(select_percentile) :: percentile
  type(select_fpr) :: fpr
  type(select_fdr) :: fdr
  type(select_fwe) :: fwe
  type(generic_univariate_select) :: generic
  real(dp) :: scores(4),pvalues(4),x(4,3),before(4,3)
  integer :: y(4),j
  real(dp) :: xt(5,1),xc(6,1),perfect(17,1)
  logical, allocatable :: mask(:)
  real(dp), allocatable :: z(:,:),restored(:,:),s(:),p(:)
  scores=[1._dp,4._dp,4._dp,0._dp]; pvalues=[.01_dp,.02_dp,.04_dp,.2_dp]
  call fpr%fit_scores(scores,pvalues); mask=fpr%get_support()
  call require(all(mask.eqv.[.true.,.true.,.true.,.false.]),'SelectFpr analytic support')
  call fdr%fit_scores(scores,pvalues); mask=fdr%get_support()
  call require(all(mask.eqv.[.true.,.true.,.false.,.false.]),'SelectFdr analytic step-up rule')
  call fwe%fit_scores(scores,pvalues); mask=fwe%get_support()
  call require(all(mask.eqv.[.true.,.false.,.false.,.false.]),'SelectFwe analytic Bonferroni rule')
  call fpr%fit_scores(scores,[.05_dp,.05_dp,1._dp,0._dp]); mask=fpr%get_support()
  call require(count(mask)==1.and.mask(4),'SelectFpr strict alpha boundary')
  call fdr%fit_scores(scores,[.0125_dp,.025_dp,.0375_dp,.05_dp]); mask=fdr%get_support()
  call require(all(mask),'SelectFdr inclusive alpha boundary')
  percentile%percentile=50; call percentile%fit_scores(scores); mask=percentile%get_support()
  call require(all(mask.eqv.[.false.,.true.,.true.,.false.]),'SelectPercentile analytic threshold')
  call percentile%fit_scores([1._dp,1._dp,1._dp,1._dp]); mask=percentile%get_support()
  call require(all(mask.eqv.[.true.,.true.,.false.,.false.]),'SelectPercentile ties in input order')
  generic%mode='k_best'; generic%param=2; call generic%fit_scores([1._dp,1._dp,1._dp,1._dp])
  mask=generic%get_support()
  call require(all(mask.eqv.[.false.,.false.,.true.,.true.]),'GenericUnivariateSelect stable top-k ties')
  generic%param=10; mask=generic%get_support()
  call require(all(mask),'GenericUnivariateSelect k above P selects all')
  generic%param=0; generic%k_all=.true.; mask=generic%get_support()
  call require(all(mask),'GenericUnivariateSelect all override')
  y=[-1,-1,1,1]; x(:,1)=real(y,dp); x(:,2)=[-1._dp,1._dp,-1._dp,1._dp]; x(:,3)=0; before=x
  call univariate_classification_scores(x,y,s,p)
  call require(.not.ieee_is_finite(s(1)).and.p(1)==0,'ANOVA perfect separation')
  call require(s(2)==0.and.p(2)==1.and.ieee_is_nan(s(3)).and.ieee_is_nan(p(3)),'ANOVA null/constant scores')
  call fpr%fit(x,y); mask=fpr%get_support()
  call require(all(mask.eqv.[.true.,.false.,.false.]),'SelectFpr built-in ANOVA')
  z=fpr%transform(x); restored=fpr%inverse_transform(z)
  call require(all(restored(:,1)==x(:,1)).and.all(restored(:,2:)==0),'univariate inverse zero fill')
  call require(all(x==before),'univariate selectors preserve inputs')
  fpr%alpha=0; z=fpr%transform(x); restored=fpr%inverse_transform(z)
  call require(size(z,2)==0.and.all(restored==0),'univariate empty selection inverse')
  call univariate_regression_scores(x,real(y,dp),s,p)
  call require(s(1)==huge(1._dp).and.p(1)==0,'regression perfect correlation')
  call require(s(3)==0.and.p(3)==1,'regression force-finite constant score')
  call univariate_classification_scores(abs(x),y,s,p,'chi2')
  call require(s(1)==0.and.p(1)==1.and.ieee_is_nan(p(3)),'chi-square null/empty counts')
  call univariate_regression_scores(x,real(y,dp)+x(:,2),s,p)
  call require(abs(s(1)-2)<1.e-13_dp.and.abs(p(1)-(1-sqrt(.5_dp)))<1.e-13_dp,'F(1,2) survival identity')
  xt(:,1)=[-2._dp,0._dp,0._dp,2._dp,0._dp]
  call univariate_classification_scores(xt,[0,0,1,1,2],s,p)
  call require(abs(s(1)-1)<1.e-13_dp.and.abs(p(1)-.5_dp)<1.e-13_dp,'F(2,2) survival symmetry')
  xc(:,1)=[3._dp,0._dp,1._dp,0._dp,2._dp,0._dp]
  call univariate_classification_scores(xc,[1,1,2,2,3,3],s,p,'chi2')
  call require(abs(s(1)-1)<1.e-13_dp.and.abs(p(1)-exp(-.5_dp))<1.e-13_dp,'chi-square two-df exponential tail')
  perfect(:,1)=[(real(j,dp),j=0,16)]
  call univariate_regression_scores(perfect,perfect(:,1),s,p)
  call require(s(1)<0.and.p(1)==1,'regression F below-support rounding endpoint')
  print *, 'Univariate statistical selection tests passed'
end program
