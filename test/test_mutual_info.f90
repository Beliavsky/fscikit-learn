program test_mutual_info
  use sklearn_mutual_info, only: mutual_info_classif,mutual_info_regression,mutual_info_score,r_regression
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  real(dp) :: x(8,3),before(8,3),y(8),small(3,3),target(3)
  real(dp), allocatable :: score(:),again(:)
  integer :: i,labels(8)
  labels=[0,0,0,0,1,1,1,1]
  x(:,1)=real(labels,dp); x(:,2)=[0._dp,1._dp,0._dp,1._dp,0._dp,1._dp,0._dp,1._dp]
  x(:,3)=0; before=x
  score=mutual_info_classif(x,labels,discrete_features=.true.)
  call require(abs(score(1)-log(2._dp))<1.e-14_dp.and.maxval(abs(score(2:)))<1.e-14_dp, &
    'MI balanced categorical dependence and independence')
  call require(abs(mutual_info_score(labels,1-labels)-log(2._dp))<1.e-14_dp,'MI cluster relabeling invariance')
  call require(mutual_info_score([1],[9])==0,'MI singleton cluster inputs')
  score=mutual_info_classif(reshape([3._dp],[1,1]),[7],discrete_features=.true.)
  call require(score(1)==0,'MI single-sample categorical features')
  score=mutual_info_classif(x,labels,n_neighbors=3,random_state=7)
  again=mutual_info_classif(x,labels,n_neighbors=3,random_state=7)
  call require(all(score==again).and.all(score>=0),'MI classification deterministic nonnegative estimates')
  do i=1,8
    y(i)=sin(real(i,dp)); x(i,1)=cos(real(i,dp)); x(i,2)=real(mod(i,2),dp)
  end do
  before=x
  score=mutual_info_regression(x,y,discrete_features=[.false.,.true.,.false.],random_state=42)
  again=mutual_info_regression(x,y,discrete_features=[2],random_state=42)
  call require(all(score==again).and.all(x==before),'MI masks, native indices and input preservation')
  again=mutual_info_regression(x,y,discrete_features=[2,2],random_state=42)
  call require(all(score==again),'MI repeated indices define one categorical feature')
  target=[-1._dp,0._dp,1._dp]; small(:,1)=target; small(:,2)=-target; small(:,3)=0
  score=r_regression(small,target)
  call require(maxval(abs(score-[1._dp,-1._dp,0._dp]))<1.e-14_dp,'R signed Pearson correlations')
  score=r_regression(small,target,force_finite=.false.)
  call require(ieee_is_nan(score(3)),'R undefined zero-feature NaN')
  score=r_regression(small,[0._dp,0._dp,0._dp])
  call require(all(score==0),'R constant target finite fallback')
  score=r_regression(reshape([1._dp,2._dp],[2,1]),[2._dp,0._dp],center=.false.)
  call require(abs(score(1)-1/sqrt(5._dp))<1.e-14_dp,'R uncentered cosine')
  score=r_regression(reshape([1._dp,1._dp],[2,1]),[1.e-200_dp,1.e-200_dp],center=.false.)
  call require(.not.ieee_is_finite(score(1)).and..not.ieee_is_nan(score(1)), &
    'R preserves source infinity on underflowed target norm')
  score=r_regression(reshape([1._dp],[1,1]),[2._dp])
  call require(score(1)==0,'R one-sample centered finite fallback')
  print *, 'Mutual information and correlation tests passed'
end program
