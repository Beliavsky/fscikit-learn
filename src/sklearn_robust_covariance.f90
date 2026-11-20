! FastMCD, MinCovDet and EllipticEnvelope: sklearn 1.6.1 covariance contracts.
! Native eigensolvers and chi-square inversion; BSD provenance retained in THIRD_PARTY_NOTICES.md.
module sklearn_robust_covariance
  use sklearn_common, only: dp,require,check_x,argsort,symmetric_eigen
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_metrics, only: accuracy_score
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_negative_inf
  implicit none
  private
  public :: min_cov_det,elliptic_envelope,fast_mcd
  type :: mcd_candidate
    real(dp), allocatable :: location(:),covariance(:,:),dist(:)
    logical, allocatable :: support(:)
    real(dp) :: determinant=-huge(1._dp)
  end type
  type :: robust_storage
    logical :: assume_centered=.false.,store_precision=.true.
    real(dp) :: support_fraction=0,correction_=1,reweight_threshold_=0
    integer :: random_state=0,n_features_in_=0,rank_=0
    real(dp), allocatable :: location_(:),covariance_(:,:),precision_(:,:),dist_(:)
    real(dp), allocatable :: raw_location_(:),raw_covariance_(:,:)
    logical, allocatable :: support_(:),raw_support_(:)
  contains
    procedure :: get_precision
    procedure :: mahalanobis
    procedure :: error_norm
  end type
  type, extends(robust_storage) :: min_cov_det
  contains
    procedure :: fit=>mcd_fit
    procedure :: score=>mcd_score
  end type
  type, extends(robust_storage) :: elliptic_envelope
    real(dp) :: contamination=.1_dp,offset_=0
  contains
    procedure :: fit=>envelope_fit
    procedure :: fit_predict=>envelope_fit_predict
    procedure :: score_samples=>envelope_samples
    procedure :: decision_function=>envelope_decision
    procedure :: predict=>envelope_predict
    procedure :: score=>envelope_score
  end type
contains
  function covariance_of(x,centered) result(covariance)
    real(dp), intent(in) :: x(:,:)
    logical, optional, intent(in) :: centered
    real(dp) :: covariance(size(x,2),size(x,2)),z(size(x,1),size(x,2)),location(size(x,2))
    call require(size(x,1)>0,'MCD covariance needs nonempty support')
    location=0
    if(present(centered)) then
      if(.not.centered) location=sum(x,dim=1)/size(x,1)
    else
      location=sum(x,dim=1)/size(x,1)
    end if
    z=x-spread(location,1,size(x,1)); covariance=matmul(transpose(z),z)/size(x,1)
  end function
  function inverse_covariance(covariance,rank) result(precision)
    real(dp), intent(in) :: covariance(:,:)
    integer, optional, intent(out) :: rank
    real(dp) :: precision(size(covariance,1),size(covariance,2))
    real(dp) :: values(size(covariance,1)),vectors(size(covariance,1),size(covariance,2)),cutoff,inverse(size(covariance,1))
    call symmetric_eigen(covariance,values,vectors)
    cutoff=epsilon(1._dp)*size(values)*maxval(abs(values)); inverse=0
    where(abs(values)>cutoff) inverse=1/merge(values,1._dp,abs(values)>cutoff)
    precision=matmul(vectors*spread(inverse,1,size(values)),transpose(vectors))
    if(present(rank)) rank=count(abs(values)>cutoff)
  end function
  real(dp) function log_determinant(covariance) result(value)
    real(dp), intent(in) :: covariance(:,:)
    real(dp) :: values(size(covariance,1)),vectors(size(covariance,1),size(covariance,2))
    call symmetric_eigen(covariance,values,vectors)
    value=-huge(1._dp)
    if(all(values>0)) value=sum(log(max(values,tiny(1._dp))))
  end function
  function distances(x,location,precision) result(values)
    real(dp), intent(in) :: x(:,:),location(:),precision(:,:)
    real(dp) :: values(size(x,1)),z(size(x,1),size(x,2))
    z=x-spread(location,1,size(x,1)); values=sum(matmul(z,precision)*z,dim=2)
  end function
  real(dp) function gamma_cdf(a,x) result(value)
    real(dp), intent(in) :: a,x
    real(dp) :: term,total,b,c,d,h,factor,an,delta
    integer :: i
    value=0
    if(x<=0) return
    factor=exp(a*log(x)-x-log_gamma(a))
    if(x<a+1) then
      term=1/a; total=term
      do i=1,10000
        term=term*x/(a+i); total=total+term
        if(abs(term)<=4*epsilon(1._dp)*abs(total)) exit
      end do
      value=total*factor
    else
      b=x+1-a; c=1.e300_dp; d=1/b; h=d
      do i=1,10000
        an=-real(i,dp)*(i-a); b=b+2; d=an*d+b
        if(abs(d)<1.e-300_dp) d=sign(1.e-300_dp,d)
        c=b+an/c
        if(abs(c)<1.e-300_dp) c=sign(1.e-300_dp,c)
        d=1/d; delta=d*c; h=h*delta
        if(abs(delta-1)<=4*epsilon(1._dp)) exit
      end do
      value=1-factor*h
    end if
    call require(i<=10000,'MCD chi-square series did not converge')
    value=max(0._dp,min(1._dp,value))
  end function
  real(dp) function chi_square_quantile(p,probability) result(value)
    integer, intent(in) :: p
    real(dp), intent(in) :: probability
    real(dp) :: low,high,middle
    integer :: i
    low=0; high=max(1._dp,real(p,dp))
    do while(gamma_cdf(real(p,dp)/2,high/2)<probability)
      high=2*high
    end do
    do i=1,100
      middle=low/2+high/2
      if(gamma_cdf(real(p,dp)/2,middle/2)<probability) then
        low=middle
      else
        high=middle
      end if
    end do
    value=low/2+high/2
  end function
  function permutation(rng,n) result(indices)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: n
    integer :: indices(n),i,j,temp
    indices=[(i,i=1,n)]
    do i=n,2,-1
      j=rng%integer_index(i); temp=indices(i); indices(i)=indices(j); indices(j)=temp
    end do
  end function
  function median(values) result(value)
    real(dp), intent(in) :: values(:)
    real(dp) :: value
    integer :: order(size(values)),n
    n=size(values); order=argsort(values)
    value=values(order((n+1)/2))/2+values(order((n+2)/2))/2
  end function
  subroutine c_step(x,h,rng,iterations,result,initial)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: h,iterations
    type(legacy_random_state), intent(inout) :: rng
    type(mcd_candidate), intent(out) :: result
    type(mcd_candidate), optional, intent(in) :: initial
    type(mcd_candidate) :: current,previous
    real(dp) :: precision(size(x,2),size(x,2)),previous_det
    real(dp), allocatable :: previous_dist(:)
    integer, allocatable :: order(:),ids(:)
    integer :: n,left
    n=size(x,1)
    if(present(initial)) then
      precision=inverse_covariance(initial%covariance)
      current%dist=distances(x,initial%location,precision); order=argsort(current%dist)
    else
      order=permutation(rng,n)
    end if
    ids=order(:h); current%location=sum(x(ids,:),dim=1)/h; current%covariance=covariance_of(x(ids,:))
    allocate(current%support(n)); current%support=.false.; current%support(ids)=.true.
    current%determinant=log_determinant(current%covariance)
    if(current%determinant==-huge(1._dp)) precision=inverse_covariance(current%covariance)
    previous_det=huge(1._dp); left=iterations
    do while(current%determinant<previous_det.and.left>0.and.current%determinant/=-huge(1._dp))
      previous=current; previous_det=current%determinant
      precision=inverse_covariance(current%covariance)
      current%dist=distances(x,current%location,precision); order=argsort(current%dist); ids=order(:h)
      current%location=sum(x(ids,:),dim=1)/h; current%covariance=covariance_of(x(ids,:))
      current%support=.false.; current%support(ids)=.true.
      current%determinant=log_determinant(current%covariance); left=left-1
    end do
    if(allocated(current%dist)) previous_dist=current%dist
    current%dist=distances(x,current%location,precision)
    result=current
    if(current%determinant/=-huge(1._dp).and.previous_det/=huge(1._dp)) then
      if(abs(current%determinant-previous_det)>1.e-8_dp+1.e-5_dp*abs(previous_det).and. &
        current%determinant>previous_det.and.left/=0) then
        result=previous; result%dist=previous_dist
      end if
    end if
  end subroutine
  subroutine select_best(x,h,rng,trials,keep,iterations,selected,initial)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: h,trials,keep,iterations
    type(legacy_random_state), intent(inout) :: rng
    type(mcd_candidate), allocatable, intent(out) :: selected(:)
    type(mcd_candidate), optional, intent(in) :: initial(:)
    type(mcd_candidate), allocatable :: candidates(:)
    real(dp), allocatable :: determinants(:)
    integer, allocatable :: order(:)
    integer :: total,j
    total=trials
    if(present(initial)) total=size(initial)
    allocate(candidates(total),determinants(total),selected(keep))
    do j=1,total
      if(present(initial)) then
        call c_step(x,h,rng,iterations,candidates(j),initial(j))
      else
        call c_step(x,h,rng,iterations,candidates(j))
      end if
      determinants(j)=candidates(j)%determinant
    end do
    order=argsort(determinants); selected=candidates(order(:keep))
  end subroutine
  subroutine fast_mcd(x,location,covariance,support,dist,support_fraction,random_state)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: location(:),covariance(:,:),dist(:)
    logical, allocatable, intent(out) :: support(:)
    real(dp), optional, intent(in) :: support_fraction
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng
    type(mcd_candidate) :: result
    type(mcd_candidate), allocatable :: initial(:),best(:),group(:),merged(:)
    real(dp) :: fraction,width,minimum,center
    real(dp), allocatable :: sorted(:),precision(:,:)
    integer, allocatable :: order(:),selection(:),shuffle(:),ids(:)
    integer :: n,p,h,seed,parts,part_size,h_part,trials,j,count_halves,merged_size,h_merged,keep
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(n>=2,'FastMCD requires at least two samples')
    fraction=0; if(present(support_fraction)) fraction=support_fraction
    call require(ieee_is_finite(fraction),'MCD support_fraction must be finite and in [0,1]')
    call require(fraction>=0.and.fraction<=1,'MCD support_fraction must be finite and in [0,1]')
    h=ceiling((real(n,dp)+p+1)/2)
    if(fraction>0) h=int(fraction*n)
    call require(h>=1.and.h<=n,'MCD support count must lie between one and n_samples')
    seed=0; if(present(random_state)) seed=random_state
    call rng%seed(seed)
    if(p==1) then
      allocate(result%location(1),result%support(n)); result%support=.false.
      if(h<n) then
        order=argsort(x(:,1)); sorted=x(order,1); minimum=huge(1._dp); center=0; count_halves=0
        do j=1,n-h
          width=sorted(j+h)-sorted(j)
          if(width<minimum) then
            minimum=width; center=sorted(j+h)+sorted(j); count_halves=1
          else if(width==minimum) then
            center=center+sorted(j+h)+sorted(j); count_halves=count_halves+1
          end if
        end do
        result%location(1)=.5_dp*center/count_halves
        order=argsort(abs(x(:,1)-result%location(1))); ids=order(:h); result%support(ids)=.true.
        result%covariance=covariance_of(x(ids,:))
      else
        result%location=sum(x,dim=1)/n; result%covariance=covariance_of(x); result%support=.true.
      end if
      precision=inverse_covariance(result%covariance); result%dist=distances(x,result%location,precision)
    else if(n>500) then
      parts=n/300; part_size=n/parts; shuffle=permutation(rng,n)
      h_part=ceiling(part_size*(real(h,dp)/n)); trials=max(10,500/parts)
      allocate(initial(parts*10))
      do j=1,parts
        selection=shuffle((j-1)*part_size+1:j*part_size)
        call select_best(x(selection,:),h_part,rng,trials,10,2,group)
        initial((j-1)*10+1:j*10)=group
      end do
      merged_size=min(1500,n); h_merged=ceiling(merged_size*(real(h,dp)/n))
      keep=1
      if(n>1500) keep=10
      order=permutation(rng,n); selection=order(:merged_size)
      call select_best(x(selection,:),h_merged,rng,0,keep,30,merged,initial)
      if(n<1500) then
        result=merged(1)
        result%support=spread(.false.,1,n); result%dist=spread(0._dp,1,n)
        result%support(selection)=merged(1)%support; result%dist(selection)=merged(1)%dist
      else
        call select_best(x,h,rng,0,1,30,best,merged); result=best(1)
      end if
    else
      call select_best(x,h,rng,30,10,2,initial)
      call select_best(x,h,rng,0,1,30,best,initial); result=best(1)
    end if
    location=result%location; covariance=result%covariance; support=result%support; dist=result%dist
  end subroutine
  subroutine robust_fit(self,x)
    class(robust_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: raw_dist(:),precision(:,:),selected(:,:)
    integer :: n,p
    call fast_mcd(x,self%raw_location_,self%raw_covariance_,self%raw_support_,raw_dist, &
      self%support_fraction,self%random_state)
    n=size(x,1); p=size(x,2); self%n_features_in_=p
    if(self%assume_centered) then
      self%raw_location_=0
      self%raw_covariance_=covariance_of(pack_rows(x,self%raw_support_),.true.)
      precision=inverse_covariance(self%raw_covariance_)
      raw_dist=distances(x,self%raw_location_,precision)
    end if
    if(count(self%raw_support_)<n) call require(any(abs(self%raw_covariance_)>1.e-8_dp), &
      'MCD raw covariance is zero; increase support_fraction')
    self%correction_=median(raw_dist)/chi_square_quantile(p,.5_dp)
    call require(ieee_is_finite(self%correction_),'MCD consistency correction must be positive and finite')
    call require(self%correction_>0,'MCD consistency correction must be positive and finite')
    self%reweight_threshold_=chi_square_quantile(p,.975_dp)
    self%support_=raw_dist/self%correction_<self%reweight_threshold_
    call require(any(self%support_),'MCD reweighting selected no samples')
    selected=pack_rows(x,self%support_); self%location_=sum(selected,dim=1)/size(selected,1)
    if(self%assume_centered) self%location_=0
    self%covariance_=covariance_of(selected,self%assume_centered)
    precision=inverse_covariance(self%covariance_,self%rank_)
    if(allocated(self%precision_)) deallocate(self%precision_)
    if(self%store_precision) self%precision_=precision
    self%dist_=distances(x,self%location_,precision)
  end subroutine
  function pack_rows(x,mask) result(rows)
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: mask(:)
    real(dp) :: rows(count(mask),size(x,2))
    integer :: i,indices(count(mask))
    indices=pack([(i,i=1,size(mask))],mask); rows=x(indices,:)
  end function
  subroutine mcd_fit(self,x)
    class(min_cov_det), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call robust_fit(self,x)
  end subroutine
  function get_precision(self) result(precision)
    class(robust_storage), intent(in) :: self
    real(dp), allocatable :: precision(:,:)
    call require(allocated(self%covariance_),'robust covariance estimator is not fitted')
    if(allocated(self%precision_)) then
      precision=self%precision_
    else
      precision=inverse_covariance(self%covariance_)
    end if
  end function
  function mahalanobis(self,x) result(values)
    class(robust_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: values(size(x,1))
    real(dp), allocatable :: precision(:,:)
    call require(allocated(self%covariance_),'robust covariance estimator is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'robust covariance feature count mismatch')
    precision=self%get_precision(); values=distances(x,self%location_,precision)
  end function
  real(dp) function mcd_score(self,x) result(value)
    class(min_cov_det), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: precision(:,:),distance(:)
    real(dp) :: determinant
    distance=self%mahalanobis(x); precision=self%get_precision(); determinant=log_determinant(precision)
    if(determinant==-huge(1._dp)) then
      value=ieee_value(0._dp,ieee_negative_inf)
    else
      value=(determinant-self%n_features_in_*log(2*acos(-1._dp))-sum(distance)/size(distance))/2
    end if
  end function
  real(dp) function error_norm(self,comp_cov,norm,scaling,squared) result(value)
    class(robust_storage), intent(in) :: self
    real(dp), intent(in) :: comp_cov(:,:)
    character(*), optional, intent(in) :: norm
    logical, optional, intent(in) :: scaling,squared
    real(dp), allocatable :: error(:,:),values(:),vectors(:,:)
    character(16) :: kind
    logical :: scale,square
    integer :: p
    call require(allocated(self%covariance_),'robust covariance estimator is not fitted')
    call require(all(shape(comp_cov)==shape(self%covariance_)),'robust covariance comparison shape mismatch')
    call check_x(comp_cov); p=self%n_features_in_; error=comp_cov-self%covariance_
    kind='frobenius'; if(present(norm)) kind=norm
    scale=.true.; if(present(scaling)) scale=scaling
    square=.true.; if(present(squared)) square=squared
    select case(trim(kind))
    case('frobenius'); value=sum(error**2)
    case('spectral')
      allocate(values(p),vectors(p,p)); call symmetric_eigen(matmul(transpose(error),error),values,vectors)
      value=maxval(values)
    case default; call require(.false.,'unknown robust covariance error norm')
    end select
    if(scale) value=value/p
    if(.not.square) value=sqrt(value)
  end function
  subroutine envelope_fit(self,x)
    class(elliptic_envelope), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    integer, allocatable :: order(:)
    real(dp) :: position
    integer :: lo,hi
    call require(ieee_is_finite(self%contamination),'EllipticEnvelope contamination must be in (0,.5]')
    call require(self%contamination>0.and.self%contamination<=.5_dp,'EllipticEnvelope contamination must be in (0,.5]')
    call robust_fit(self,x); values=-self%dist_; order=argsort(values)
    position=1+(size(values)-1)*self%contamination; lo=floor(position); hi=ceiling(position)
    self%offset_=values(order(lo))+(position-lo)*(values(order(hi))-values(order(lo)))
  end subroutine
  function envelope_samples(self,x) result(values)
    class(elliptic_envelope), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: values(size(x,1))
    values=-self%mahalanobis(x)
  end function
  function envelope_decision(self,x) result(values)
    class(elliptic_envelope), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: values(size(x,1))
    values=self%score_samples(x)-self%offset_
  end function
  function envelope_predict(self,x) result(labels)
    class(elliptic_envelope), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    labels=merge(1,-1,self%decision_function(x)>=0)
  end function
  function envelope_fit_predict(self,x) result(labels)
    class(elliptic_envelope), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%predict(x)
  end function
  real(dp) function envelope_score(self,x,y,sample_weight) result(value)
    class(elliptic_envelope), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer :: labels(size(x,1))
    labels=self%predict(x)
    value=accuracy_score(y,labels)
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'EllipticEnvelope score weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'EllipticEnvelope score weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'EllipticEnvelope score weights must be finite and nonnegative')
      call require(sum(sample_weight)>0,'EllipticEnvelope score weights must have positive sum')
      value=sum(sample_weight,mask=labels==y)/sum(sample_weight)
    end if
  end function
end module
