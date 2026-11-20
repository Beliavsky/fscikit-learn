! Univariate selectors and score contracts: sklearn 1.6.1
! feature_selection/_univariate_selection.py, BSD-3-Clause.
! Upstream notice retained in LICENSES/scikit-learn-COPYING.
module sklearn_univariate_selection
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,argsort
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  private
  public :: select_percentile,select_fpr,select_fdr,select_fwe,generic_univariate_select
  public :: univariate_classification_scores,univariate_regression_scores
  type :: generic_univariate_select
    real(dp) :: alpha=.05_dp,percentile=10,param=1.e-5_dp
    integer :: n_features_in_=0
    character(16) :: mode='percentile',score_func='auto'
    logical :: center=.true.,k_all=.false.
    real(dp), allocatable :: scores_(:),pvalues_(:)
  contains
    procedure :: fit_classification=>selector_class_fit
    procedure :: fit_regression=>selector_reg_fit
    generic :: fit=>fit_classification,fit_regression
    procedure :: fit_scores=>selector_scores_fit
    procedure :: get_support=>selector_support
    procedure :: get_support_indices=>selector_indices
    procedure :: transform=>selector_transform
    procedure :: inverse_transform=>selector_inverse
  end type
  type, extends(generic_univariate_select) :: select_percentile
  end type
  type, extends(generic_univariate_select) :: select_fpr
  end type
  type, extends(generic_univariate_select) :: select_fdr
  end type
  type, extends(generic_univariate_select) :: select_fwe
  end type
contains
  real(dp) function beta_fraction(a,b,x) result(h)
    ! DLMF 8.17.22-23 beta fraction, evaluated by modified Lentz iteration.
    real(dp), intent(in) :: a,b,x
    real(dp), parameter :: floor_value=1.e-300_dp
    real(dp) :: c,d,delta,aa,m2
    integer :: m
    c=1; d=1-(a+b)*x/(a+1)
    if(abs(d)<floor_value) d=sign(floor_value,d)
    d=1/d; h=d
    do m=1,20000
      m2=2._dp*m; aa=m*(b-m)*x/((a-1+m2)*(a+m2))
      d=1+aa*d; c=1+aa/c
      if(abs(d)<floor_value) d=sign(floor_value,d)
      if(abs(c)<floor_value) c=sign(floor_value,c)
      d=1/d; h=h*d*c
      aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+1+m2))
      d=1+aa*d; c=1+aa/c
      if(abs(d)<floor_value) d=sign(floor_value,d)
      if(abs(c)<floor_value) c=sign(floor_value,c)
      d=1/d; delta=d*c; h=h*delta
      if(abs(delta-1)<8*epsilon(1._dp)) exit
    end do
    call require(m<=20000,'univariate beta fraction failed to converge')
  end function
  real(dp) function f_tail(value,df1,df2) result(probability)
    real(dp), intent(in) :: value
    integer, intent(in) :: df1,df2
    real(dp) :: a,b,x,y,ratio,prefactor
    probability=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(value).or.df1<=0.or.df2<=0) return
    if(value<0) return
    if(value==0) then
      probability=1; return
    end if
    if(.not.ieee_is_finite(value)) then
      probability=0; return
    end if
    a=.5_dp*df2; b=.5_dp*df1
    if(value>real(df2,dp)/df1) then
      ratio=(real(df2,dp)/df1)/value; x=ratio/(1+ratio); y=1/(1+ratio)
    else
      ratio=(real(df1,dp)/df2)*value; x=1/(1+ratio); y=ratio/(1+ratio)
    end if
    if(x==0) then
      probability=0; return
    end if
    if(y==0) then
      probability=1; return
    end if
    prefactor=exp(log_gamma(a+b)-log_gamma(a)-log_gamma(b)+a*log(x)+b*log(y))
    if(x<(a+1)/(a+b+2)) then
      probability=prefactor*beta_fraction(a,b,x)/a
    else
      probability=1-prefactor*beta_fraction(b,a,y)/b
    end if
    probability=max(0._dp,min(1._dp,probability))
  end function
  real(dp) function chi_tail(value,df) result(probability)
    ! DLMF 8.8.6 upper gamma recurrence; positive terms in log space.
    real(dp), intent(in) :: value
    integer, intent(in) :: df
    real(dp) :: x,shape
    integer :: k
    probability=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(value).or.df<1) return
    if(value<0) return
    if(value==0) then
      probability=1; return
    end if
    if(.not.ieee_is_finite(value)) then
      probability=0; return
    end if
    x=.5_dp*value; probability=0
    if(mod(df,2)==1) then
      probability=erfc(sqrt(x))
      do k=0,df/2-1
        shape=k+.5_dp
        probability=probability+exp(shape*log(x)-x-log_gamma(shape+1))
      end do
    else
      do k=0,df/2-1
        probability=probability+exp(real(k,dp)*log(x)-x-log_gamma(real(k+1,dp)))
      end do
    end if
    probability=max(0._dp,min(1._dp,probability))
  end function

  subroutine univariate_classification_scores(x,y,scores,pvalues,score_func)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable, intent(out) :: scores(:),pvalues(:)
    character(*), optional, intent(in) :: score_func
    integer, allocatable :: labels(:)
    real(dp), allocatable :: totals(:),sums(:),squares(:),between(:),within(:),expected(:)
    real(dp) :: nan_value
    integer :: c,j,n,k,nc
    character(16) :: kind
    kind='f_classif'; if(present(score_func)) kind=score_func
    call require(kind=='f_classif'.or.kind=='chi2','unknown univariate classification score')
    call check_xy(x,size(y)); n=size(y); labels=unique_labels(y); k=size(labels)
    allocate(scores(size(x,2)),pvalues(size(x,2)))
    nan_value=ieee_value(0._dp,ieee_quiet_nan); scores=nan_value; pvalues=nan_value
    if(kind=='chi2') then
      call require(all(x>=0),'univariate chi2 requires nonnegative X')
      if(k<2) return
      totals=sum(x,dim=1); scores=0
      do c=1,k
        sums=matmul(real(merge(1,0,y==labels(c)),dp),x)
        expected=totals*(real(count(y==labels(c)),dp)/n)
        do j=1,size(x,2)
          if(expected(j)>0) then
            scores(j)=scores(j)+(sums(j)-expected(j))**2/expected(j)
          else
            scores(j)=nan_value
          end if
        end do
      end do
      do j=1,size(scores)
        pvalues(j)=chi_tail(scores(j),k-1)
      end do
    else
      if(k<2.or.n<=k) return
      totals=sum(x,dim=1); squares=0*totals; between=0*totals
      do c=1,k
        nc=count(y==labels(c)); sums=0*totals
        do j=1,n
          if(y(j)/=labels(c)) cycle
          sums=sums+x(j,:); squares=squares+x(j,:)**2
        end do
        between=between+sums**2/real(nc,dp)
      end do
      within=squares-totals**2/real(n,dp)
      between=between-totals**2/real(n,dp); within=within-between
      do j=1,size(scores)
        if(within(j)==0) then
          if(between(j)/=0) scores(j)=sign(ieee_value(0._dp,ieee_positive_inf),between(j))
        else
          scores(j)=(between(j)/real(k-1,dp))/(within(j)/real(n-k,dp))
        end if
        pvalues(j)=f_tail(scores(j),k-1,n-k)
      end do
    end if
  end subroutine

  subroutine univariate_regression_scores(x,y,scores,pvalues,center)
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable, intent(out) :: scores(:),pvalues(:)
    logical, optional, intent(in) :: center
    real(dp) :: target(size(y)),norm_x,norm_y,r,r2,mean
    logical :: centered
    integer :: j,n,df
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    centered=.true.; if(present(center)) centered=center
    n=size(y); df=n-merge(2,1,centered)
    call require(df>0,'univariate regression requires positive residual degrees of freedom')
    allocate(scores(size(x,2)),pvalues(size(x,2))); target=y
    if(centered) target=y-sum(y)/real(n,dp)
    norm_y=sqrt(sum(target**2))
    do j=1,size(x,2)
      mean=0; if(centered) mean=sum(x(:,j))/real(n,dp)
      norm_x=sqrt(max(0._dp,sum(x(:,j)**2)-real(n,dp)*mean**2))
      if(norm_x==0.or.norm_y==0) then
        scores(j)=0; pvalues(j)=1; cycle
      end if
      r=dot_product(target,x(:,j))/norm_x/norm_y; r2=r*r
      if(r2==1) then
        scores(j)=huge(1._dp); pvalues(j)=0
      else
        scores(j)=r2/(1-r2)*df
        if(scores(j)<0) then
          ! stats.f.sf returns one below support; ANOVA uses special.fdtrc.
          pvalues(j)=1
        else
          pvalues(j)=f_tail(scores(j),1,df)
        end if
      end if
    end do
  end subroutine

  subroutine selector_class_fit(self,x,y,score_func)
    class(generic_univariate_select), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    character(*), optional, intent(in) :: score_func
    real(dp), allocatable :: scores(:),pvalues(:)
    character(16) :: kind
    kind=self%score_func; if(kind=='auto') kind='f_classif'
    if(present(score_func)) kind=score_func
    call univariate_classification_scores(x,y,scores,pvalues,kind)
    call self%fit_scores(scores,pvalues)
  end subroutine
  subroutine selector_reg_fit(self,x,y)
    class(generic_univariate_select), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: scores(:),pvalues(:)
    call require(self%score_func=='auto'.or.self%score_func=='f_regression','unknown univariate regression score')
    call univariate_regression_scores(x,y,scores,pvalues,self%center)
    call self%fit_scores(scores,pvalues)
  end subroutine
  subroutine selector_scores_fit(self,scores,pvalues)
    class(generic_univariate_select), intent(inout) :: self
    real(dp), intent(in) :: scores(:)
    real(dp), optional, intent(in) :: pvalues(:)
    logical, allocatable :: mask(:)
    call require(size(scores)>0,'univariate scores must be nonempty')
    self%scores_=scores; self%n_features_in_=size(scores)
    if(allocated(self%pvalues_)) deallocate(self%pvalues_)
    if(present(pvalues)) then
      call require(size(pvalues)==size(scores),'univariate p-value count mismatch')
      self%pvalues_=pvalues
    end if
    mask=self%get_support()
  end subroutine
  subroutine selection_control(self,kind,value)
    class(generic_univariate_select), intent(in) :: self
    character(*), intent(out) :: kind
    real(dp), intent(out) :: value
    select type(self)
    type is(select_percentile); kind='percentile'; value=self%percentile
    type is(select_fpr); kind='fpr'; value=self%alpha
    type is(select_fdr); kind='fdr'; value=self%alpha
    type is(select_fwe); kind='fwe'; value=self%alpha
    class default; kind=self%mode; value=self%param
    end select
    if(kind=='k_best'.and.self%k_all) value=real(self%n_features_in_,dp)
    call require(ieee_is_finite(value),'univariate selection parameter must be finite')
    select case(trim(kind))
    case('percentile'); call require(value>=0.and.value<=100,'univariate percentile must be in [0,100]')
    case('fpr','fdr','fwe'); call require(value>=0.and.value<=1,'univariate alpha must be in [0,1]')
    case('k_best')
      call require(value>=0.and.value<=real(huge(1),dp),'univariate k must be a nonnegative supported integer')
      call require(value==aint(value),'univariate k must be integral')
    case default; call require(.false.,'unknown univariate selection mode')
    end select
  end subroutine
  function selector_support(self) result(mask)
    class(generic_univariate_select), intent(in) :: self
    logical, allocatable :: mask(:)
    real(dp), allocatable :: clean(:)
    integer, allocatable :: order(:),ties(:),ids(:)
    real(dp) :: value,threshold,position,fraction,lower,upper
    character(16) :: kind
    integer :: p,j,left,right,keep
    call require(allocated(self%scores_),'univariate selector is not fitted')
    call selection_control(self,kind,value); p=size(self%scores_); allocate(mask(p)); mask=.false.
    if(kind=='percentile'.or.kind=='k_best') then
      clean=self%scores_
      do j=1,p
        if(ieee_is_nan(clean(j))) clean(j)=-huge(1._dp)
      end do
      order=argsort(clean)
      if(kind=='k_best') then
        keep=min(p,int(value)); if(keep>0) mask(order(p-keep+1:))=.true.
      else if(value==100) then
        mask=.true.
      else if(value/=0) then
        position=real(p-1,dp)*((100-value)/100); left=int(floor(position))+1; right=min(p,left+1)
        fraction=position-floor(position); lower=clean(order(left)); upper=clean(order(right))
        threshold=ieee_value(0._dp,ieee_quiet_nan)
        if(ieee_is_finite(lower).and.ieee_is_finite(upper)) then
          if(lower<0.and.upper>0) then
            if(upper>huge(1._dp)+lower) then
              if(fraction>0.and.fraction<1) then
                threshold=ieee_value(0._dp,ieee_positive_inf)
                if(fraction>=.5_dp) threshold=-threshold
              end if
            else if(fraction<.5_dp) then
              threshold=lower+(upper-lower)*fraction
            else
              threshold=upper-(upper-lower)*(1-fraction)
            end if
          else if(fraction<.5_dp) then
            threshold=lower+(upper-lower)*fraction
          else
            threshold=upper-(upper-lower)*(1-fraction)
          end if
        else if(lower==upper) then
          ! NumPy interpolation subtracts equal infinities, yielding NaN.
        else if(fraction>0.and.fraction<1) then
          if(fraction<.5_dp.and.ieee_is_finite(lower)) then
            threshold=upper
          else if(fraction>=.5_dp.and.ieee_is_finite(upper)) then
            threshold=lower
          end if
        end if
        if(.not.ieee_is_nan(threshold)) then
          mask=clean>threshold; ids=[(j,j=1,p)]; ties=pack(ids,clean==threshold)
          keep=int(real(p,dp)*value/100)-count(mask)
          if(keep<0) keep=size(ties)+keep
          keep=max(0,min(size(ties),keep)); mask(ties(:keep))=.true.
        end if
      end if
    else
      call require(allocated(self%pvalues_),'univariate selection mode requires p-values')
      select case(kind)
      case('fpr','fwe')
        threshold=value; if(kind=='fwe') threshold=value/p
        do j=1,p
          if(ieee_is_nan(self%pvalues_(j))) cycle
          mask(j)=self%pvalues_(j)<threshold
        end do
      case('fdr')
        clean=self%pvalues_
        do j=1,p
          if(ieee_is_nan(clean(j))) clean(j)=ieee_value(0._dp,ieee_positive_inf)
        end do
        order=argsort(clean); threshold=-huge(1._dp)
        do j=1,p
          if(clean(order(j))<=value/real(p,dp)*j) threshold=clean(order(j))
        end do
        do j=1,p
          if(ieee_is_nan(self%pvalues_(j))) cycle
          mask(j)=self%pvalues_(j)<=threshold
        end do
      end select
    end if
  end function
  function selector_indices(self) result(indices)
    class(generic_univariate_select), intent(in) :: self
    integer, allocatable :: indices(:),ids(:)
    logical, allocatable :: mask(:)
    integer :: j
    mask=self%get_support(); ids=[(j,j=1,size(mask))]; indices=pack(ids,mask)
  end function
  function selector_transform(self,x) result(z)
    class(generic_univariate_select), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer, allocatable :: indices(:)
    indices=self%get_support_indices(); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'univariate transform feature count mismatch')
    z=x(:,indices)
  end function
  function selector_inverse(self,x) result(z)
    class(generic_univariate_select), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer, allocatable :: indices(:)
    indices=self%get_support_indices()
    call require(size(x,1)>0,'univariate inverse requires nonempty rows')
    call require(all(ieee_is_finite(x)),'univariate inverse contains nonfinite values')
    call require(size(x,2)==size(indices),'univariate inverse feature count mismatch')
    allocate(z(size(x,1),self%n_features_in_)); z=0; z(:,indices)=x
  end function
end module
