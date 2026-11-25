! Dummy baselines adapted from sklearn 1.6.1 dummy.py (BSD-3-Clause).
! NumPy legacy RNG/binomial helpers retain their existing notices.
module sklearn_dummy
  use sklearn_common,only:dp,require,unique_labels,label_index
  use sklearn_numpy_ordering,only:argsort=>numpy_argsort
  use sklearn_metrics,only:accuracy_score,r2_score
  use sklearn_sparse,only:csr_matrix
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_random_binomial,only:legacy_binomial
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_is_nan,ieee_next_after
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::dummy_classifier,dummy_regressor,dummy_class_distribution,dummy_probability
  type::dummy_class_distribution
    integer,allocatable::classes(:)
    real(dp),allocatable::prior(:)
  end type
  type::dummy_probability
    real(dp),allocatable::values(:,:)
  end type
  type::dummy_classifier
    character(16)::strategy='prior'
    character(16),private::fitted_strategy_=''
    integer::constant=0,random_state=0,n_classes_=0,n_outputs_=0,n_features_in_=0
    integer,allocatable::constant_values(:),classes_(:),n_classes_multi_(:)
    real(dp),allocatable::class_prior_(:)
    type(dummy_class_distribution),allocatable::distributions_(:)
  contains
    procedure,private::class_fit_dv
    procedure,private::class_fit_dv_scalar
    procedure,private::class_fit_dm
    procedure,private::class_fit_dm_scalar
    procedure,private::class_fit_sv
    procedure,private::class_fit_sv_scalar
    procedure,private::class_fit_sm
    procedure,private::class_fit_sm_scalar
    generic::fit=>class_fit_dv,class_fit_dv_scalar,class_fit_dm,class_fit_dm_scalar, &
      class_fit_sv,class_fit_sv_scalar,class_fit_sm,class_fit_sm_scalar
    procedure,private::class_predict_d
    procedure,private::class_predict_multi_d
    procedure,private::class_score_v_d
    procedure,private::class_score_m_d
    procedure,private::class_proba_d
    procedure,private::class_proba_multi_d
    procedure,private::class_predict_s
    procedure,private::class_predict_multi_s
    procedure,private::class_score_v_s
    procedure,private::class_score_m_s
    procedure,private::class_proba_s
    procedure,private::class_proba_multi_s
    generic::predict=>class_predict_d,class_predict_s
    generic::predict_multi=>class_predict_multi_d,class_predict_multi_s
    generic::predict_proba=>class_proba_d,class_proba_s
    generic::predict_proba_multi=>class_proba_multi_d,class_proba_multi_s
    generic::score=>class_score_v_d,class_score_m_d,class_score_v_s,class_score_m_s
  end type
  type::dummy_regressor
    character(16)::strategy='mean'
    real(dp)::constant=0,quantile=.5_dp,constant_=0
    real(dp),allocatable::constant_values(:),constants_(:)
    integer::n_outputs_=0,n_features_in_=0
    logical::fitted_=.false.
  contains
    procedure,private::reg_fit_dv
    procedure,private::reg_fit_dv_scalar
    procedure,private::reg_fit_dm
    procedure,private::reg_fit_dm_scalar
    procedure,private::reg_fit_sv
    procedure,private::reg_fit_sv_scalar
    procedure,private::reg_fit_sm
    procedure,private::reg_fit_sm_scalar
    generic::fit=>reg_fit_dv,reg_fit_dv_scalar,reg_fit_dm,reg_fit_dm_scalar, &
      reg_fit_sv,reg_fit_sv_scalar,reg_fit_sm,reg_fit_sm_scalar
    procedure,private::reg_predict_d
    procedure,private::reg_predict_multi_d
    procedure,private::reg_score_v_d
    procedure,private::reg_score_m_d
    procedure,private::reg_predict_s
    procedure,private::reg_predict_multi_s
    procedure,private::reg_score_v_s
    procedure,private::reg_score_m_s
    generic::predict=>reg_predict_d,reg_predict_s
    generic::predict_multi=>reg_predict_multi_d,reg_predict_multi_s
    generic::score=>reg_score_v_d,reg_score_m_d,reg_score_v_s,reg_score_m_s
  end type
contains
  subroutine fp_begin(modes,flags)
    logical,intent(out)::modes(:),flags(:)
    call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
  end subroutine
  subroutine fp_end(modes,flags)
    logical,intent(in)::modes(:),flags(:)
    call ieee_set_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_all,flags)
  end subroutine
  subroutine weights_for(n,weights,w)
    integer,intent(in)::n
    real(dp),optional,intent(in)::weights(:)
    real(dp),allocatable,intent(out)::w(:)
    allocate(w(n));w=1
    if(present(weights))then
      call require(size(weights)==n,'dummy weight count mismatch')
      call require(all(ieee_is_finite(weights)),'dummy weights must be finite')
      w=weights
    end if
  end subroutine
  subroutine class_fit_core(self,n,p,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    integer,intent(in)::n,p,y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::w(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::j,c,i,constant
    call require(n==size(y,1).and.n>0.and.size(y,2)>0,'dummy classifier target shape mismatch')
    call require(any(self%strategy==[character(16)::'prior','most_frequent','stratified','uniform','constant']), &
      'unknown dummy classifier strategy')
    call require(self%random_state>=0,'dummy random_state must be nonnegative')
    self%n_outputs_=size(y,2);self%n_features_in_=p
    if(allocated(self%distributions_))deallocate(self%distributions_)
    allocate(self%distributions_(self%n_outputs_))
    self%n_classes_multi_=spread(0,1,self%n_outputs_)
    if(self%strategy=='constant')then
      if(allocated(self%constant_values))then
        call require(size(self%constant_values)==self%n_outputs_,'dummy classifier constant count mismatch')
      else
        call require(self%n_outputs_==1,'multioutput dummy classifier requires constant_values')
      end if
    end if
    call weights_for(n,sample_weight,w);call fp_begin(modes,flags)
    do j=1,self%n_outputs_
      associate(head=>self%distributions_(j))
        head%classes=unique_labels(y(:,j));self%n_classes_multi_(j)=size(head%classes)
        head%prior=spread(0._dp,1,size(head%classes))
        do i=1,n
          c=label_index(head%classes,y(i,j));head%prior(c)=head%prior(c)+w(i)
        end do
        head%prior=head%prior/sum(head%prior)
        if(self%strategy=='constant')then
          constant=self%constant
          if(allocated(self%constant_values))constant=self%constant_values(j)
          call require(label_index(head%classes,constant)>0,'unknown dummy classifier constant')
        end if
      end associate
    end do
    self%classes_=self%distributions_(1)%classes;self%class_prior_=self%distributions_(1)%prior
    self%n_classes_=size(self%classes_);self%fitted_strategy_=self%strategy
    call fp_end(modes,flags)
  end subroutine
  function class_proba_core(self,n)result(out)
    class(dummy_classifier),intent(in)::self
    integer,intent(in)::n
    type(dummy_probability),allocatable::out(:)
    type(legacy_random_state)::rng
    real(dp)::remaining,p
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::j,k,i,c,chosen,constant
    call require(allocated(self%distributions_),'dummy classifier is not fitted')
    allocate(out(self%n_outputs_));call rng%seed(self%random_state)
    call fp_begin(modes,flags)
    do j=1,self%n_outputs_
      associate(head=>self%distributions_(j))
        k=size(head%classes);allocate(out(j)%values(n,k));out(j)%values=0
        select case(self%fitted_strategy_)
        case('prior');out(j)%values=spread(head%prior,1,n)
        case('uniform');out(j)%values=1._dp/k
        case('most_frequent');out(j)%values(:,first_argmax(head%prior))=1
        case('constant')
          constant=self%constant
          if(allocated(self%constant_values))then
            call require(size(self%constant_values)==self%n_outputs_,'dummy classifier constant count mismatch')
            constant=self%constant_values(j)
          end if
          c=label_index(head%classes,constant)
          call require(c>0,'unknown dummy classifier prediction constant');out(j)%values(:,c)=1
        case('stratified')
          call require(all(ieee_is_finite(head%prior)).and.all(head%prior>=0).and.all(head%prior<=1), &
            'stratified dummy requires valid probabilities')
          do i=1,n
            remaining=1;chosen=k
            do c=1,k-1
              p=head%prior(c)/remaining
              if(legacy_binomial(1,p,rng)==1)then
                chosen=c;exit
              end if
              remaining=remaining-head%prior(c)
            end do
            out(j)%values(i,chosen)=1
          end do
        end select
      end associate
    end do
    call fp_end(modes,flags)
  end function
  integer function first_argmax(x)result(index)
    real(dp),intent(in)::x(:)
    integer::j
    index=1
    do j=2,size(x)
      ! NumPy argmax selects its first NaN, including zero-total priors.
      if(.not.ieee_is_finite(x(index)))then
        if(ieee_is_nan(x(index)))return
      end if
      if(ieee_is_nan(x(j)))then
        index=j;return
      end if
      if(x(j)>x(index))index=j
    end do
  end function
  function class_predict_core(self,n)result(y)
    class(dummy_classifier),intent(in)::self
    integer,intent(in)::n
    integer,allocatable::y(:,:)
    type(dummy_probability),allocatable::proba(:)
    type(legacy_random_state)::rng
    integer::j,i,c
    call require(allocated(self%distributions_),'dummy classifier is not fitted')
    allocate(y(n,self%n_outputs_))
    if(self%fitted_strategy_=='uniform')then
      call rng%seed(self%random_state)
      do j=1,self%n_outputs_
        do i=1,n
          c=rng%integer_index(size(self%distributions_(j)%classes));y(i,j)=self%distributions_(j)%classes(c)
        end do
      end do
    else
      proba=class_proba_core(self,n)
      do j=1,self%n_outputs_
        do i=1,n
          y(i,j)=self%distributions_(j)%classes(first_argmax(proba(j)%values(i,:)))
        end do
      end do
    end if
  end function
  subroutine reg_fit_core(self,n,p,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    integer,intent(in)::n,p
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::w(:),cdf(:)
    integer,allocatable::order(:)
    real(dp)::q,position,target
    integer::j,i,lo,hi,mid
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    self%fitted_=.false.
    call require(n==size(y,1).and.n>0.and.size(y,2)>0,'dummy regressor target shape mismatch')
    call require(all(ieee_is_finite(y)),'dummy regressor target must be finite')
    call require(any(self%strategy==[character(16)::'mean','median','quantile','constant']), &
      'unknown dummy regressor strategy')
    q=self%quantile;if(self%strategy=='median')q=.5_dp
    call require(ieee_is_finite(self%quantile),'dummy quantile must be finite')
    call require(self%quantile>=0.and.self%quantile<=1,'invalid dummy quantile')
    self%n_outputs_=size(y,2);self%n_features_in_=p
    self%constants_=spread(0._dp,1,self%n_outputs_)
    if(self%strategy=='constant')then
      if(allocated(self%constant_values))then
        call require(size(self%constant_values)==self%n_outputs_,'dummy regressor constant count mismatch')
        call require(all(ieee_is_finite(self%constant_values)),'dummy regression constants must be finite')
      else
        call require(self%n_outputs_==1,'multioutput dummy regressor requires constant_values')
        call require(ieee_is_finite(self%constant),'dummy regression constant must be finite')
      end if
    end if
    call weights_for(n,sample_weight,w);call fp_begin(modes,flags)
    do j=1,self%n_outputs_
      select case(self%strategy)
      case('mean')
        call require(sum(w)/=0,'dummy mean requires nonzero total weight')
        self%constants_(j)=sum(y(:,j)*w)/sum(w)
      case('constant')
        self%constants_(j)=self%constant
        if(allocated(self%constant_values))self%constants_(j)=self%constant_values(j)
      case('median','quantile')
        order=argsort(y(:,j))
        if(present(sample_weight))then
          cdf=w(order)
          do i=2,n
            cdf(i)=cdf(i)+cdf(i-1)
          end do
          target=q*cdf(n)
          if(target==0)target=ieee_next_after(target,target+1)
          lo=1;hi=n+1
          do while(lo<hi)
            mid=lo+(hi-lo)/2
            if(cdf(mid)<target)then
              lo=mid+1
            else
              hi=mid
            end if
          end do
          self%constants_(j)=y(order(min(lo,n)),j)
        else
          position=(n-1)*q;lo=floor(position)+1;hi=ceiling(position)+1
          self%constants_(j)=y(order(lo),j)+(position-floor(position))*(y(order(hi),j)-y(order(lo),j))
        end if
      end select
    end do
    self%constant_=self%constants_(1);self%fitted_=.true.;call fp_end(modes,flags)
  end subroutine
  function reg_predict_core(self,n)result(y)
    class(dummy_regressor),intent(in)::self
    integer,intent(in)::n
    real(dp),allocatable::y(:,:)
    call require(self%fitted_,'dummy regressor is not fitted')
    y=spread(self%constants_,1,n)
  end function
  subroutine class_fit_dv(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call class_fit_core(self,size(x,1),size(x,2),reshape(y,[size(y),1]),sample_weight)
  end subroutine
  subroutine class_fit_dv_scalar(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),intent(in)::sample_weight
    call class_fit_core(self,size(x,1),size(x,2),reshape(y,[size(y),1]),spread(sample_weight,1,size(x,1)))
  end subroutine
  real(dp) function class_score_v_d(self,x,y,sample_weight)result(value)
    class(dummy_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  subroutine class_fit_dm(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call class_fit_core(self,size(x,1),size(x,2),y,sample_weight)
  end subroutine
  subroutine class_fit_dm_scalar(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    real(dp),intent(in)::sample_weight
    call class_fit_core(self,size(x,1),size(x,2),y,spread(sample_weight,1,size(x,1)))
  end subroutine
  real(dp) function class_score_m_d(self,x,y,sample_weight)result(value)
    class(dummy_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    if(size(y,2)==1)then
      value=accuracy_score(y(:,1),self%predict(x),sample_weight=sample_weight)
    else
      if(size(y,2)==1)then
      value=accuracy_score(y(:,1),self%predict(x),sample_weight=sample_weight)
    else
      value=accuracy_score(y,self%predict_multi(x),sample_weight=sample_weight)
    end if
    end if
  end function
  function class_predict_d(self,x)result(y)
    class(dummy_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,allocatable::y(:)
    integer,allocatable::matrix(:,:)
    call require(self%n_outputs_==1,'vector dummy prediction requires one output')
    matrix=class_predict_core(self,size(x,1));y=matrix(:,1)
  end function
  function class_predict_multi_d(self,x)result(y)
    class(dummy_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,allocatable::y(:,:)
    y=class_predict_core(self,size(x,1))
  end function
  function class_proba_d(self,x)result(out)
    class(dummy_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::out(:,:)
    type(dummy_probability),allocatable::heads(:)
    call require(self%n_outputs_==1,'matrix dummy probabilities require one output')
    heads=class_proba_core(self,size(x,1));out=heads(1)%values
  end function
  function class_proba_multi_d(self,x)result(out)
    class(dummy_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    type(dummy_probability),allocatable::out(:)
    out=class_proba_core(self,size(x,1))
  end function
  subroutine class_fit_sv(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call x%validate()
    call class_fit_core(self,x%n_rows,x%n_cols,reshape(y,[size(y),1]),sample_weight)
  end subroutine
  subroutine class_fit_sv_scalar(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),intent(in)::sample_weight
    call x%validate()
    call class_fit_core(self,x%n_rows,x%n_cols,reshape(y,[size(y),1]),spread(sample_weight,1,x%n_rows))
  end subroutine
  real(dp) function class_score_v_s(self,x,y,sample_weight)result(value)
    class(dummy_classifier),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  subroutine class_fit_sm(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call x%validate()
    call class_fit_core(self,x%n_rows,x%n_cols,y,sample_weight)
  end subroutine
  subroutine class_fit_sm_scalar(self,x,y,sample_weight)
    class(dummy_classifier),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:,:)
    real(dp),intent(in)::sample_weight
    call x%validate()
    call class_fit_core(self,x%n_rows,x%n_cols,y,spread(sample_weight,1,x%n_rows))
  end subroutine
  real(dp) function class_score_m_s(self,x,y,sample_weight)result(value)
    class(dummy_classifier),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=accuracy_score(y,self%predict_multi(x),sample_weight=sample_weight)
  end function
  function class_predict_s(self,x)result(y)
    class(dummy_classifier),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::y(:)
    integer,allocatable::matrix(:,:)
    call x%validate()
    call require(self%n_outputs_==1,'vector dummy prediction requires one output')
    matrix=class_predict_core(self,x%n_rows);y=matrix(:,1)
  end function
  function class_predict_multi_s(self,x)result(y)
    class(dummy_classifier),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::y(:,:)
    call x%validate()
    y=class_predict_core(self,x%n_rows)
  end function
  function class_proba_s(self,x)result(out)
    class(dummy_classifier),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::out(:,:)
    type(dummy_probability),allocatable::heads(:)
    call x%validate()
    call require(self%n_outputs_==1,'matrix dummy probabilities require one output')
    heads=class_proba_core(self,x%n_rows);out=heads(1)%values
  end function
  function class_proba_multi_s(self,x)result(out)
    class(dummy_classifier),intent(in)::self
    type(csr_matrix),intent(in)::x
    type(dummy_probability),allocatable::out(:)
    call x%validate()
    out=class_proba_core(self,x%n_rows)
  end function
  subroutine reg_fit_dv(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call reg_fit_core(self,size(x,1),size(x,2),reshape(y,[size(y),1]),sample_weight)
  end subroutine
  subroutine reg_fit_dv_scalar(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),intent(in)::sample_weight
    call require(.false.,'dummy regressor requires vector sample weights, as in sklearn 1.6.1')
  end subroutine
  real(dp) function reg_score_v_d(self,x,y,sample_weight)result(value)
    class(dummy_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  subroutine reg_fit_dm(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call reg_fit_core(self,size(x,1),size(x,2),y,sample_weight)
  end subroutine
  subroutine reg_fit_dm_scalar(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),intent(in)::sample_weight
    call require(.false.,'dummy regressor requires vector sample weights, as in sklearn 1.6.1')
  end subroutine
  real(dp) function reg_score_m_d(self,x,y,sample_weight)result(value)
    class(dummy_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict_multi(x),sample_weight=sample_weight)
  end function
  function reg_predict_d(self,x)result(y)
    class(dummy_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:)
    real(dp),allocatable::matrix(:,:)
    call require(self%n_outputs_==1,'vector dummy prediction requires one output')
    matrix=reg_predict_core(self,size(x,1));y=matrix(:,1)
  end function
  function reg_predict_multi_d(self,x)result(y)
    class(dummy_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:,:)
    y=reg_predict_core(self,size(x,1))
  end function
  subroutine reg_fit_sv(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call x%validate()
    call reg_fit_core(self,x%n_rows,x%n_cols,reshape(y,[size(y),1]),sample_weight)
  end subroutine
  subroutine reg_fit_sv_scalar(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),intent(in)::sample_weight
    call x%validate()
    call require(.false.,'dummy regressor requires vector sample weights, as in sklearn 1.6.1')
  end subroutine
  real(dp) function reg_score_v_s(self,x,y,sample_weight)result(value)
    class(dummy_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  subroutine reg_fit_sm(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call x%validate()
    call reg_fit_core(self,x%n_rows,x%n_cols,y,sample_weight)
  end subroutine
  subroutine reg_fit_sm_scalar(self,x,y,sample_weight)
    class(dummy_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),intent(in)::sample_weight
    call x%validate()
    call require(.false.,'dummy regressor requires vector sample weights, as in sklearn 1.6.1')
  end subroutine
  real(dp) function reg_score_m_s(self,x,y,sample_weight)result(value)
    class(dummy_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict_multi(x),sample_weight=sample_weight)
  end function
  function reg_predict_s(self,x)result(y)
    class(dummy_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:)
    real(dp),allocatable::matrix(:,:)
    call x%validate()
    call require(self%n_outputs_==1,'vector dummy prediction requires one output')
    matrix=reg_predict_core(self,x%n_rows);y=matrix(:,1)
  end function
  function reg_predict_multi_s(self,x)result(y)
    class(dummy_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:,:)
    call x%validate()
    y=reg_predict_core(self,x%n_rows)
  end function
end module
