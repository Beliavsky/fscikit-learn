! Bayesian chained imputation following sklearn 1.6.1 impute/_iterative.py.
! BSD-3-Clause attribution is retained in THIRD_PARTY_NOTICES.md.
module sklearn_iterative_impute
  use sklearn_common, only: dp,require,argsort
  use sklearn_linear_model, only: bayesian_ridge
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_mean,ieee_difference,ieee_quotient
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  private
  public :: iterative_imputer,imputer_triplet
  type :: imputer_triplet
    integer :: feat_idx=0
    integer, allocatable :: neighbor_feat_idx(:)
    type(bayesian_ridge) :: estimator
  end type
  type :: iterative_imputer
    integer :: max_iter=10,n_nearest_features=0,random_state=0
    real(dp) :: tol=1.e-3_dp,fill_value=0,missing_values=0
    real(dp) :: min_value=-huge(1._dp),max_value=huge(1._dp)
    real(dp), allocatable :: min_values(:),max_values(:)
    character(16) :: initial_strategy='mean',imputation_order='ascending'
    logical :: skip_complete=.false.,keep_empty_features=.false.,add_indicator=.false.
    logical :: sample_posterior=.false.,missing_is_nan=.true.,converged_=.false.
    integer :: n_features_in_=0,n_features_with_missing_=0,n_iter_=0
    type(bayesian_ridge) :: estimator
    type(imputer_triplet), allocatable :: imputation_sequence_(:)
    real(dp), allocatable :: initial_statistics_(:)
    integer, allocatable :: valid_features_(:),indicator_features_(:)
    integer, allocatable, private :: name_features_(:)
    logical, allocatable, private :: empty_(:)
    real(dp), allocatable, private :: lower_(:),upper_(:)
    type(legacy_random_state), private :: rng_
    logical, private :: seeded_=.false.,indicator_enabled_=.false.
  contains
    procedure :: fit=>iterative_fit
    procedure :: fit_transform=>iterative_fit_transform
    procedure :: transform=>iterative_transform
    procedure :: get_feature_names_out=>iterative_names
  end type
contains
  function missing_mask(self,x) result(mask)
    class(iterative_imputer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical :: mask(size(x,1),size(x,2))
    call require(size(x,1)>0.and.size(x,2)>0,'IterativeImputer X must be nonempty')
    if(self%missing_is_nan) then
      call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'IterativeImputer X contains infinity')
      mask=ieee_is_nan(x)
    else
      call require(ieee_is_finite(self%missing_values),'IterativeImputer missing sentinel must be finite')
      call require(all(ieee_is_finite(x)),'IterativeImputer finite-sentinel input contains NaN or infinity')
      mask=x==self%missing_values
    end if
  end function
  subroutine controls(self)
    class(iterative_imputer), intent(in) :: self
    call require(self%max_iter>=0,'IterativeImputer max_iter must be nonnegative')
    call require(ieee_is_finite(self%tol),'IterativeImputer tol must be finite')
    call require(self%tol>=0,'IterativeImputer tol must be nonnegative')
    call require(self%n_nearest_features>=0,'IterativeImputer n_nearest_features must be nonnegative')
    call require(self%random_state>=0,'IterativeImputer random_state must be nonnegative')
    call require(any(self%initial_strategy==[character(16)::'mean','median','most_frequent','constant']), &
      'IterativeImputer unknown initial strategy')
    call require(any(self%imputation_order==[character(16)::'ascending','descending','roman','arabic','random']), &
      'IterativeImputer unknown imputation order')
    call require(ieee_is_finite(self%fill_value),'IterativeImputer fill_value must be finite')
  end subroutine
  function initialize(self,x,mask) result(z)
    class(iterative_imputer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical, intent(out), allocatable :: mask(:,:)
    real(dp), allocatable :: z(:,:)
    logical, allocatable :: original(:,:)
    integer :: j,k
    original=missing_mask(self,x)
    call require(size(x,2)==self%n_features_in_,'IterativeImputer feature count mismatch')
    z=x(:,self%valid_features_); mask=original(:,self%valid_features_)
    do j=1,size(z,2)
      k=self%valid_features_(j)
      where(mask(:,j)) z(:,j)=self%initial_statistics_(k)
      if(self%empty_(k)) mask(:,j)=.false.
    end do
  end function
  function feature_order(self,mask) result(order)
    class(iterative_imputer), intent(inout) :: self
    logical, intent(in) :: mask(:,:)
    integer, allocatable :: order(:),all_indices(:)
    real(dp) :: fractions(size(mask,2))
    integer :: j,k,temp
    fractions=real(count(mask,dim=1),dp)/size(mask,1)
    all_indices=[(j,j=1,size(mask,2))]
    if(self%imputation_order=='ascending'.or.self%imputation_order=='descending') all_indices=argsort(fractions)
    if(self%skip_complete) then
      order=pack(all_indices,fractions(all_indices)>0)
    else
      order=all_indices
    end if
    if(self%imputation_order=='arabic'.or.self%imputation_order=='descending') order=order(size(order):1:-1)
    if(self%imputation_order=='random') then
      do j=size(order),2,-1
        k=self%rng_%integer_index(j); temp=order(j); order(j)=order(k); order(k)=temp
      end do
    end if
  end function
  function correlations(z) result(corr)
    real(dp), intent(in) :: z(:,:)
    real(dp) :: corr(size(z,2),size(z,2)),centered(size(z,1),size(z,2)),norms(size(z,2)),v
    integer :: j,k
    do j=1,size(z,2)
      centered(:,j)=z(:,j)-ieee_mean(z(:,j)); norms(j)=sqrt(sum(centered(:,j)**2))
    end do
    corr=1.e-6_dp
    do j=1,size(z,2)
      do k=1,size(z,2)
        if(norms(j)>0.and.norms(k)>0) then
          v=abs(dot_product(centered(:,j)/norms(j),centered(:,k)/norms(k)))
          corr(k,j)=max(1.e-6_dp,min(1._dp,v))
        end if
      end do
      corr(j,j)=0; corr(:,j)=corr(:,j)/sum(corr(:,j))
    end do
  end function
  function neighbors(self,p,feature,corr) result(indices)
    class(iterative_imputer), intent(inout) :: self
    integer, intent(in) :: p,feature
    real(dp), intent(in) :: corr(:,:)
    integer, allocatable :: indices(:)
    real(dp) :: probabilities(p),cdf(p),draws(p),u
    integer :: needed,filled,j,k,chosen
    if(self%n_nearest_features==0.or.self%n_nearest_features>=p) then
      indices=pack([(j,j=1,p)],[(j/=feature,j=1,p)]); return
    end if
    allocate(indices(self%n_nearest_features)); probabilities=corr(:,feature); filled=0
    ! NumPy weighted choice without replacement draws a batch, keeps the first
    ! occurrence of each new index, then draws the unfilled remainder again.
    do while(filled<size(indices))
      if(filled>0) probabilities(indices(:filled))=0
      cdf=probabilities
      do j=2,p
        cdf(j)=cdf(j)+cdf(j-1)
      end do
      cdf=cdf/cdf(p); needed=size(indices)-filled
      do j=1,needed
        draws(j)=self%rng_%uniform()
      end do
      do j=1,needed
        u=draws(j); chosen=p
        do k=1,p
          if(u<cdf(k)) then
            chosen=k; exit
          end if
        end do
        if(any(indices(:filled)==chosen)) cycle
        filled=filled+1; indices(filled)=chosen
      end do
    end do
  end function
  real(dp) function center_quantile(q) result(value)
    real(dp), intent(in) :: q
    real(dp) :: lo,hi,mid,tail,delta,pi
    integer :: j
    pi=acos(-1._dp)
    if(abs(q)<.1_dp) then
      value=q*sqrt(2*pi)
      do j=1,6
        delta=(.5_dp*erf(value/sqrt(2._dp))-q)*sqrt(2*pi)*exp(.5_dp*value*value)
        value=value-delta
      end do
    else
      tail=.5_dp-abs(q); lo=-40; hi=0
      do j=1,80
        mid=.5_dp*(lo+hi)
        if(.5_dp*erfc(-mid/sqrt(2._dp))<tail) then
          lo=mid
        else
          hi=mid
        end if
      end do
      value=sign(-.5_dp*(lo+hi),q)
    end if
  end function
  subroutine impute_feature(self,z,mask,triplet,training)
    class(iterative_imputer), intent(inout) :: self
    real(dp), intent(inout) :: z(:,:)
    logical, intent(in) :: mask(:,:)
    type(imputer_triplet), intent(inout) :: triplet
    logical, intent(in) :: training
    integer, allocatable :: observed(:),missing(:)
    real(dp), allocatable :: x(:,:),prediction(:),sigma(:)
    real(dp) :: low,high,a,b,q,u,v
    integer :: j,k,n
    k=triplet%feat_idx; n=size(z,1)
    observed=pack([(j,j=1,n)],.not.mask(:,k)); missing=pack([(j,j=1,n)],mask(:,k))
    if(training) then
      triplet%estimator=self%estimator
      call triplet%estimator%fit(z(observed,triplet%neighbor_feat_idx),z(observed,k))
    end if
    if(size(missing)==0) return
    x=z(missing,triplet%neighbor_feat_idx); prediction=triplet%estimator%predict(x)
    call require(all(ieee_is_finite(prediction)),'IterativeImputer prediction exceeds finite working range')
    low=self%lower_(k); high=self%upper_(k)
    if(self%sample_posterior) then
      sigma=triplet%estimator%predict_std(x)
      call require(all(ieee_is_finite(sigma)),'IterativeImputer posterior uncertainty exceeds finite working range')
      do j=1,size(missing)
        v=prediction(j)
        if(v<low) then
          v=low
        else if(v>high) then
          v=high
        else if(sigma(j)>0) then
          a=ieee_quotient(ieee_difference(low,v),sigma(j))
          b=ieee_quotient(ieee_difference(high,v),sigma(j))
          a=.5_dp*erf(a/sqrt(2._dp)); b=.5_dp*erf(b/sqrt(2._dp))
          u=self%rng_%uniform(); q=(1-u)*a+u*b
          v=v+sigma(j)*center_quantile(q)
        end if
        prediction(j)=max(low,min(high,v))
      end do
    else
      prediction=max(low,min(high,prediction))
    end if
    z(missing,k)=prediction
  end subroutine
  function concatenate(self,z,original_mask) result(output)
    class(iterative_imputer), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    logical, intent(in) :: original_mask(:,:)
    real(dp), allocatable :: output(:,:)
    integer :: p,q
    p=size(z,2); q=0; if(self%indicator_enabled_) q=size(self%indicator_features_)
    allocate(output(size(z,1),p+q)); output(:,:p)=z
    if(q>0) output(:,p+1:)=merge(1._dp,0._dp,original_mask(:,self%indicator_features_))
  end function
  function iterative_fit_transform(self,x) result(output)
    class(iterative_imputer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: output(:,:),z(:,:),previous(:,:),values(:),corr(:,:),limits(:)
    logical, allocatable :: original(:,:),mask(:,:)
    integer, allocatable :: order(:),sorted(:)
    real(dp) :: threshold,change,best
    integer :: j,k,p,n,start,run,used,round,idx
    type(imputer_triplet), allocatable :: sequence(:)
    call controls(self); original=missing_mask(self,x); p=size(x,2); n=size(x,1)
    self%n_features_in_=p; self%empty_=all(original,dim=1)
    self%valid_features_=pack([(j,j=1,p)],.not.self%empty_.or.self%keep_empty_features)
    self%name_features_=self%valid_features_
    ! sklearn 1.6.1's constant initial imputer retains empty-column names even
    ! though IterativeImputer itself drops those data columns (upstream #29827).
    if(self%initial_strategy=='constant') self%name_features_=[(j,j=1,p)]
    self%indicator_features_=pack([(j,j=1,p)],any(original,dim=1)); self%indicator_enabled_=self%add_indicator
    if(allocated(self%initial_statistics_)) deallocate(self%initial_statistics_)
    allocate(self%initial_statistics_(p)); self%initial_statistics_=0
    do j=1,p
      values=pack(x(:,j),.not.original(:,j)); k=size(values)
      if(self%initial_strategy=='constant') then
        self%initial_statistics_(j)=self%fill_value
      else if(k>0) then
        if(self%initial_strategy=='mean') then
          self%initial_statistics_(j)=ieee_mean(values)
        else
          sorted=argsort(values); values=values(sorted)
          if(self%initial_strategy=='median') then
            self%initial_statistics_(j)=ieee_mean(values((k+1)/2:k/2+1))
          else
            best=values(1); run=1; start=1
            do idx=2,k
              if(values(idx)==values(idx-1)) then
                run=run+1
              else
                run=1
              end if
              if(run>start) then
                best=values(idx); start=run
              end if
            end do
            self%initial_statistics_(j)=best
          end if
        end if
      end if
    end do
    z=initialize(self,x,mask); p=size(z,2)
    if(allocated(self%imputation_sequence_)) deallocate(self%imputation_sequence_)
    allocate(self%imputation_sequence_(0)); self%n_iter_=0; self%n_features_with_missing_=0; self%converged_=.false.
    if(.not.self%seeded_) then
      call self%rng_%seed(self%random_state); self%seeded_=.true.
    end if
    if(self%max_iter==0.or.all(mask).or.p<=1) then
      output=concatenate(self,z,original); return
    end if
    limits=spread(self%min_value,1,self%n_features_in_)
    if(allocated(self%min_values)) then
      call require(size(self%min_values)==self%n_features_in_,'IterativeImputer min_values width mismatch')
      limits=self%min_values
    end if
    self%lower_=limits(self%valid_features_); limits=self%max_value
    if(allocated(self%max_values)) then
      call require(size(self%max_values)==self%n_features_in_,'IterativeImputer max_values width mismatch')
      limits=self%max_values
    end if
    self%upper_=limits(self%valid_features_)
    call require(.not.any(ieee_is_nan(self%lower_)).and..not.any(ieee_is_nan(self%upper_)), &
      'IterativeImputer bounds contain NaN')
    call require(all(self%lower_<self%upper_),'IterativeImputer min_value must be below max_value')
    order=feature_order(self,mask); self%n_features_with_missing_=size(order)
    call require(int(size(order),int64)*self%max_iter<huge(1),'IterativeImputer sequence exceeds integer capacity')
    allocate(sequence(size(order)*self%max_iter)); used=0
    allocate(corr(p,p)); corr=0
    if(self%n_nearest_features>0.and.self%n_nearest_features<p) corr=correlations(z)
    previous=z; threshold=0
    if(.not.self%sample_posterior) threshold=self%tol*maxval(abs(pack(z,.not.mask)))
    do round=1,self%max_iter
      if(self%imputation_order=='random') order=feature_order(self,mask)
      do j=1,size(order)
        used=used+1; k=order(j); sequence(used)%feat_idx=k
        sequence(used)%neighbor_feat_idx=neighbors(self,p,k,corr)
        call impute_feature(self,z,mask,sequence(used),.true.)
      end do
      self%n_iter_=round
      if(.not.self%sample_posterior) then
        change=maxval(sum(abs(z-previous),dim=2))
        if(change<threshold) then
          self%converged_=.true.; exit
        end if
        previous=z
      end if
    end do
    self%imputation_sequence_=sequence(:used); output=concatenate(self,z,original)
  end function
  subroutine iterative_fit(self,x)
    class(iterative_imputer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: output(:,:)
    output=self%fit_transform(x)
  end subroutine
  function iterative_transform(self,x) result(output)
    class(iterative_imputer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: output(:,:),z(:,:)
    logical, allocatable :: mask(:,:),original(:,:)
    type(imputer_triplet) :: triplet
    integer :: j
    call require(allocated(self%imputation_sequence_),'IterativeImputer is not fitted')
    original=missing_mask(self,x); z=initialize(self,x,mask)
    if(self%n_iter_>0.and..not.all(mask)) then
      do j=1,size(self%imputation_sequence_)
        triplet=self%imputation_sequence_(j)
        call impute_feature(self,z,mask,triplet,.false.)
      end do
    end if
    output=concatenate(self,z,original)
  end function
  function iterative_names(self,input_features) result(names)
    class(iterative_imputer), intent(in) :: self
    character(*), optional, intent(in) :: input_features(:)
    character(128), allocatable :: names(:),base(:)
    integer :: j,p,q
    call require(allocated(self%imputation_sequence_),'IterativeImputer is not fitted')
    allocate(base(self%n_features_in_))
    do j=1,size(base)
      write(base(j),'(a,i0)') 'x',j-1
    end do
    if(present(input_features)) then
      call require(size(input_features)==self%n_features_in_,'IterativeImputer feature-name width mismatch')
      base=input_features
    end if
    p=size(self%name_features_); q=0; if(self%indicator_enabled_) q=size(self%indicator_features_)
    allocate(names(p+q)); names(:p)=base(self%name_features_)
    do j=1,q
      names(p+j)='missingindicator_'//trim(base(self%indicator_features_(j)))
    end do
  end function
end module
