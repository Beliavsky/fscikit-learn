! SGDRegressor: sklearn 1.6.1 _stochastic_gradient.py and _sgd_fast.pyx.tp.
! Lazy weight scaling also follows sklearn utils/_weight_vector.pyx.tp.
! BSD attribution: LICENSES/scikit-learn-COPYING; THIRD_PARTY_NOTICES.md.
module sklearn_sgd
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index,sigmoid
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse, only: csr_matrix
  use sklearn_resampling, only: shuffle_split,stratified_shuffle_split,index_split
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_metrics, only: r2_score,accuracy_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_negative_inf
  implicit none
  private
  public :: sgd_regressor,sgd_classifier
  type :: sgd_regressor
    character(32) :: loss='squared_error',penalty='l2',learning_rate='invscaling'
    real(dp) :: alpha=1.e-4_dp,eta0=.01_dp,power_t=.25_dp,epsilon=.1_dp,tol=1.e-3_dp
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,shuffle=.false.,converged_=.false.
    real(dp) :: intercept_=0,t_=1,loss_=0
    real(dp), allocatable :: coef_(:)
    real(dp), allocatable, private :: standard_coef(:),average_coef(:)
    real(dp), private :: standard_intercept=0,average_intercept=0
    logical, private :: classification=.false.
    integer, private :: fitted_average=0
    real(dp) :: l1_ratio=.15_dp
    logical :: warm_start=.false.,early_stopping=.false.
    real(dp) :: validation_fraction=.1_dp
    logical,allocatable :: validation_mask_(:)
  contains
    procedure, private :: fit_dense=>sgd_fit,fit_sparse=>sgd_fit_sparse
    generic :: fit=>fit_dense,fit_sparse
    procedure, private :: partial_dense=>sgd_partial,partial_sparse=>sgd_partial_sparse
    generic :: partial_fit=>partial_dense,partial_sparse
    procedure, private :: predict_dense=>sgd_predict,predict_sparse=>sgd_predict_sparse
    generic :: predict=>predict_dense,predict_sparse
    procedure, private :: score_dense=>sgd_score,score_sparse=>sgd_score_sparse
    generic :: score=>score_dense,score_sparse
  end type
  ! Classification wrappers use one signed binary SGD head per OVR channel.
  type :: sgd_classifier
    character(32) :: loss='hinge',penalty='l2',learning_rate='invscaling'
    real(dp) :: alpha=1.e-4_dp,eta0=.01_dp,power_t=.25_dp,tol=1.e-3_dp,t_=1
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,shuffle=.false.,converged_=.false.,class_weight_balanced=.false.
    integer, allocatable :: classes_(:),n_iter_per_class_(:)
    logical, allocatable :: converged_per_class_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:),loss_(:)
    type(sgd_regressor), allocatable, private :: heads(:)
    real(dp) :: l1_ratio=.15_dp
    real(dp) :: epsilon=.1_dp
    logical :: warm_start=.false.,early_stopping=.false.
    real(dp) :: validation_fraction=.1_dp
    logical,allocatable :: validation_mask_(:)
  contains
    procedure, private :: fit_matrix=>sgdc_fit,fit_vector=>sgdc_fit_vector
    procedure, private :: fit_scalar_bias=>sgdc_fit_scalar_bias,fit_vector_scalar_bias=>sgdc_fit_vector_scalar_bias
    procedure, private :: fit_csr=>sgdc_fit_sparse,fit_csr_vector=>sgdc_fit_sparse_vector
    procedure, private :: fit_csr_scalar=>sgdc_fit_sparse_scalar,fit_csr_vector_scalar=>sgdc_fit_sparse_vector_scalar
    generic :: fit=>fit_matrix,fit_vector,fit_scalar_bias,fit_vector_scalar_bias, &
      fit_csr,fit_csr_vector,fit_csr_scalar,fit_csr_vector_scalar
    procedure, private :: partial_fit_dense=>sgdc_partial,partial_fit_sparse=>sgdc_partial_sparse
    generic :: partial_fit=>partial_fit_dense,partial_fit_sparse
    procedure, private :: decision_function_dense=>sgdc_decision,decision_function_sparse=>sgdc_decision_sparse
    generic :: decision_function=>decision_function_dense,decision_function_sparse
    procedure, private :: predict_dense=>sgdc_predict,predict_sparse=>sgdc_predict_sparse
    generic :: predict=>predict_dense,predict_sparse
    procedure, private :: predict_proba_dense=>sgdc_proba,predict_proba_sparse=>sgdc_proba_sparse
    generic :: predict_proba=>predict_proba_dense,predict_proba_sparse
    procedure, private :: predict_log_proba_dense=>sgdc_log_proba,predict_log_proba_sparse=>sgdc_log_proba_sparse
    generic :: predict_log_proba=>predict_log_proba_dense,predict_log_proba_sparse
    procedure, private :: score_dense=>sgdc_score,score_sparse=>sgdc_score_sparse
    generic :: score=>score_dense,score_sparse
  end type
contains
  function make_validation_mask(n,fraction,seed,weights,labels)result(mask)
    integer,intent(in)::n,seed
    real(dp),intent(in)::fraction,weights(:)
    integer,optional,intent(in)::labels(:)
    logical::mask(n)
    type(shuffle_split)::random
    type(stratified_shuffle_split)::stratified
    type(index_split),allocatable::folds(:)
    if(present(labels))then
      stratified%n_splits=1;stratified%test_size=fraction;stratified%random_state=seed
      folds=stratified%split(labels)
    else
      random%n_splits=1;random%test_size=fraction;random%random_state=seed
      folds=random%split(n)
    end if
    call require(any(weights(folds(1)%test)>0),'SGD validation sample weights are all zero')
    mask=.false.;mask(folds(1)%test)=.true.
  end function
  subroutine sgd_controls(self)
    class(sgd_regressor), intent(in) :: self
    if(self%classification) then
      call require(any(self%loss==[character(32)::'hinge','squared_hinge','perceptron', &
        'log_loss','modified_huber','squared_error','huber','epsilon_insensitive', &
        'squared_epsilon_insensitive']),'unknown SGD classification loss')
    else
      call require(any(self%loss==[character(32)::'squared_error','huber', &
        'epsilon_insensitive','squared_epsilon_insensitive']),'unknown SGD regression loss')
    end if
    call require(any(self%penalty==[character(32)::'none','l1','l2','elasticnet']),'unsupported SGD penalty')
    call require(any(self%learning_rate==[character(32)::'constant','invscaling','optimal','adaptive']), &
      'unsupported SGD learning rate')
    call require(all(ieee_is_finite([self%alpha,self%eta0,self%power_t,self%epsilon,self%tol])),'nonfinite SGD controls')
    call require(self%alpha>=0,'SGD alpha must be nonnegative')
    call require(self%eta0>0.or.(self%eta0==0.and.self%learning_rate=='optimal'),'SGD eta0 must be positive')
    if(self%learning_rate=='optimal') call require(self%alpha>0,'optimal SGD requires positive alpha')
    call require(ieee_is_finite(self%l1_ratio),'SGD l1_ratio must be finite')
    call require(self%l1_ratio>=0.and.self%l1_ratio<=1,'SGD l1_ratio must be in [0,1]')
    call require(self%random_state>=0,'SGD random_state must be nonnegative')
    call require(self%power_t>=0,'SGD power_t must be nonnegative')
    call require(self%epsilon>=0,'SGD epsilon must be nonnegative')
    call require(self%tol>=0.or.self%tol== -1,'SGD tol must be nonnegative or -1')
    call require(self%max_iter>0.and.self%n_iter_no_change>0,'invalid SGD iteration controls')
    call require(self%average>=0,'SGD average must be nonnegative')
    call require(ieee_is_finite(self%validation_fraction),'nonfinite SGD validation_fraction')
    call require(self%validation_fraction>0.and.self%validation_fraction<1,'SGD validation_fraction must lie in (0,1)')
  end subroutine
  subroutine sgd_initialize(self,p)
    class(sgd_regressor), intent(inout) :: self
    integer, intent(in) :: p
    self%standard_coef=spread(0._dp,1,p); self%average_coef=self%standard_coef; self%coef_=self%standard_coef
    self%standard_intercept=0; self%average_intercept=0; self%intercept_=0; self%t_=1
    self%n_features_in_=p; self%fitted_average=self%average
  end subroutine
  subroutine sgd_fit(self,x,y,sample_weight,coef_init,intercept_init)
    class(sgd_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), intent(in), optional :: coef_init(:),intercept_init
    call sgd_controls(self); call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call sgd_prepare_fit(self,size(x,2),coef_init,intercept_init)
    call sgd_run(self,x,y,self%max_iter,.true.,sample_weight)
  end subroutine
  subroutine sgd_prepare_fit(self,p,coef_init,intercept_init)
    class(sgd_regressor), intent(inout) :: self
    integer, intent(in) :: p
    real(dp), optional, intent(in) :: coef_init(:),intercept_init
    if(self%warm_start.and.allocated(self%coef_)) then
      call require(p==self%n_features_in_,'SGD warm_start feature count mismatch')
      call require(self%average==self%fitted_average,'SGD averaging cannot change during warm_start')
      if(self%average==0) then
        self%standard_coef=self%coef_;self%standard_intercept=self%intercept_
      end if
      self%t_=1
      ! sklearn's regressor ignores fit initializers on an already warm fit.
      ! Averaged warm fits retain standard and accumulated-average buffers.
    else
      call sgd_initialize(self,p)
      if(present(coef_init)) then
        call require(size(coef_init)==p,'SGD coefficient initializer shape mismatch')
        call require(all(ieee_is_finite(coef_init)),'SGD coefficient initializer must be finite')
        self%standard_coef=coef_init
      end if
      if(present(intercept_init)) then
        call require(ieee_is_finite(intercept_init),'SGD intercept initializer must be finite')
        self%standard_intercept=intercept_init
      end if
    end if
  end subroutine
  subroutine check_csr(x,n)
    type(csr_matrix), intent(in) :: x
    integer, optional, intent(in) :: n
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'X must be nonempty')
    call require(all(ieee_is_finite(x%data)),'X contains NaN or infinity')
    if(present(n)) call require(x%n_rows==n,'X and y sample counts differ')
  end subroutine
  subroutine sgd_fit_sparse(self,x,y,sample_weight,coef_init,intercept_init)
    class(sgd_regressor), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),coef_init(:),intercept_init
    real(dp) :: empty(0,0)
    call sgd_controls(self);call check_csr(x,size(y));call check_x(reshape(y,[size(y),1]))
    call sgd_prepare_fit(self,x%n_cols,coef_init,intercept_init)
    call sgd_run(self,empty,y,self%max_iter,.true.,sample_weight,x)
  end subroutine
  subroutine sgd_partial_sparse(self,x,y,sample_weight)
    class(sgd_regressor), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: empty(0,0)
    call sgd_controls(self);call check_csr(x,size(y));call check_x(reshape(y,[size(y),1]))
    call require(.not.self%early_stopping,'early_stopping must be false with partial_fit')
    if(.not.allocated(self%standard_coef)) call sgd_initialize(self,x%n_cols)
    call require(x%n_cols==self%n_features_in_,'SGD feature count mismatch')
    call require(self%average==self%fitted_average,'SGD averaging cannot change during partial_fit')
    call sgd_run(self,empty,y,1,.false.,sample_weight,x)
  end subroutine
  subroutine sgd_partial(self,x,y,sample_weight)
    class(sgd_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call sgd_controls(self); call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(.not.self%early_stopping,'early_stopping must be false with partial_fit')
    if(.not.allocated(self%standard_coef)) call sgd_initialize(self,size(x,2))
    call require(size(x,2)==self%n_features_in_,'SGD feature count mismatch')
    call require(self%average==self%fitted_average,'SGD averaging cannot change during partial_fit')
    call sgd_run(self,x,y,1,.false.,sample_weight)
  end subroutine
  subroutine sgd_loss(self,prediction,target,value,gradient)
    class(sgd_regressor), intent(in) :: self
    real(dp), intent(in) :: prediction,target
    real(dp), intent(out) :: value,gradient
    real(dp) :: excess,residual,margin,e
    residual=prediction-target; margin=prediction*target
    select case(self%loss)
    case('hinge','perceptron')
      excess=1
      if(self%loss=='perceptron') excess=0
      value=max(0._dp,excess-margin); gradient=0
      if(margin<=excess) gradient=-target
    case('squared_hinge')
      excess=max(0._dp,1-margin); value=excess**2; gradient=-2*target*excess
    case('modified_huber')
      if(margin>=1) then
        value=0; gradient=0
      else if(margin>= -1) then
        value=(1-margin)**2; gradient=-2*target*(1-margin)
      else
        value=-4*margin; gradient=-4*target
      end if
    case('log_loss')
      if(margin>=0) then
        e=exp(-margin); value=log(1+e); gradient=-target*e/(1+e)
      else
        e=exp(margin); value=-margin+log(1+e); gradient=-target/(1+e)
      end if
    case('squared_error'); value=.5_dp*residual**2; gradient=residual
    case('huber')
      if(abs(residual)<=self%epsilon) then
        value=.5_dp*residual**2; gradient=residual
      else
        value=self%epsilon*(abs(residual)-.5_dp*self%epsilon); gradient=sign(self%epsilon,residual)
      end if
    case('epsilon_insensitive')
      excess=max(0._dp,abs(residual)-self%epsilon); value=excess; gradient=0
      if(excess>0) gradient=sign(1._dp,residual)
    case('squared_epsilon_insensitive')
      excess=max(0._dp,abs(residual)-self%epsilon); value=excess**2; gradient=2*sign(excess,residual)
    end select
  end subroutine
  subroutine sgd_run(self,x,y,epochs,stopping,sample_weight,sparse_x,validation_mask,validation_weight)
    class(sgd_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer, intent(in) :: epochs
    logical, intent(in) :: stopping
    real(dp), intent(in), optional :: sample_weight(:)
    type(csr_matrix), optional, intent(in) :: sparse_x
    logical,optional,intent(in)::validation_mask(:)
    real(dp),optional,intent(in)::validation_weight(:)
    logical::held_out(size(y))
    integer,allocatable::validation_indices(:)
    real(dp),allocatable::validation_prediction(:),score_weight(:)
    real(dp)::best_score,validation_score,start_t
    real(dp) :: intercept_decay
    real(dp) :: weights(size(y)),sumloss,best_loss,value,gradient,eta,update,count,optimal_init,typw
    real(dp) :: work(size(self%standard_coef)),wscale,inner,q(size(self%standard_coef)),u,ratio,z
    real(dp) :: average_a,average_b,mu
    type(legacy_random_state) :: rng
    integer(int64) :: state
    integer :: order(size(y)),n,i,j,it,pick,tmp,stale,previous_stale,feature,seed,k,lo,hi,entry
    intercept_decay=1;if(present(sparse_x)) intercept_decay=.01_dp
    n=size(y); weights=1; work=self%standard_coef; wscale=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'SGD sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite SGD sample weights')
      call require(all(sample_weight>=0),'negative SGD sample weights'); weights=sample_weight
    end if
    held_out=.false.;start_t=self%t_
    if(self%early_stopping)then
      if(present(validation_mask))then
        held_out=validation_mask
      else
        held_out=make_validation_mask(n,self%validation_fraction,self%random_state,weights)
      end if
      validation_indices=pack([(i,i=1,n)],held_out)
      allocate(validation_prediction(size(validation_indices)))
      score_weight=weights(validation_indices)
      if(present(validation_weight))score_weight=validation_weight(validation_indices)
    end if
    self%validation_mask_=held_out
    best_score=ieee_value(0._dp,ieee_negative_inf)
    order=[(i,i=1,n)]; best_loss=huge(1._dp); stale=0; self%converged_=.false.
    call rng%seed(self%random_state)
    if(self%classification) tmp=rng%integer_index(2147483646)
    seed=rng%integer_index(2147483647)-1
    eta=self%eta0;optimal_init=0;u=0;q=0;average_a=0;average_b=1;ratio=self%l1_ratio
    if(self%penalty=='l2') ratio=0
    if(self%penalty=='l1') ratio=1
    if(self%learning_rate=='optimal') then
      typw=sqrt(1/sqrt(self%alpha))
      call sgd_loss(self,-typw,1._dp,value,gradient)
      optimal_init=1/((typw/max(1._dp,gradient))*self%alpha)
    end if
    do it=1,epochs
      if(self%shuffle) then
        state=int(seed,int64)
        do j=1,n-1
          pick=j+modulo(next_random(state),n-j+1);tmp=order(j);order(j)=order(pick);order(pick)=tmp
        end do
      end if
      sumloss=0
      do j=1,n
        i=order(j)
        if(held_out(i))cycle
        inner=0
        if(present(sparse_x)) then
          lo=sparse_x%indptr(i);hi=sparse_x%indptr(i+1)-1
          do k=lo,hi
            feature=sparse_x%indices(k);inner=inner+work(feature)*sparse_x%data(k)
          end do
        else
          do feature=1,size(work)
            inner=inner+work(feature)*x(i,feature)
          end do
        end if
        call sgd_loss(self,inner*wscale+self%standard_intercept,y(i),value,gradient)
        if(.not.self%early_stopping)sumloss=sumloss+value
        if(self%learning_rate=='optimal') eta=1/(self%alpha*(optimal_init+self%t_-1))
        if(self%learning_rate=='invscaling') eta=self%eta0/self%t_**self%power_t
        gradient=max(-1.e12_dp,min(1.e12_dp,gradient)); update=-eta*gradient*weights(i)
        ! Match source WeightVector64 lazy L2 scaling and reset threshold.
        if(self%penalty=='l2'.or.self%penalty=='elasticnet') wscale=wscale*max(0._dp,1-((1-ratio)*eta*self%alpha))
        if(wscale<1.e-9_dp) then
          call reset_scale()
        end if
        if(update/=0) then
          if(present(sparse_x)) then
            do k=lo,hi
              feature=sparse_x%indices(k);work(feature)=work(feature)+sparse_x%data(k)*(update/wscale)
            end do
          else
            work=work+x(i,:)*(update/wscale)
          end if
        end if
        if(self%fit_intercept) self%standard_intercept=self%standard_intercept+update*intercept_decay
        if(self%average>0.and.self%t_>=self%average) then
          count=self%t_-self%average+1
          mu=1/count
          if(present(sparse_x)) then
            do k=lo,hi
              feature=sparse_x%indices(k)
              self%average_coef(feature)=self%average_coef(feature)+average_a*sparse_x%data(k)*(-update/wscale)
            end do
          else
            self%average_coef=self%average_coef+average_a*x(i,:)*(-update/wscale)
          end if
          if(count>1) average_b=average_b/(1-mu)
          average_a=average_a+mu*average_b*wscale
          self%average_intercept=self%average_intercept+(self%standard_intercept-self%average_intercept)/count
        end if
        if(self%penalty=='l1'.or.self%penalty=='elasticnet') then
          u=u+ratio*eta*self%alpha
          lo=1;hi=size(work)
          if(present(sparse_x)) then
            lo=sparse_x%indptr(i);hi=sparse_x%indptr(i+1)-1
          end if
          do entry=lo,hi
            feature=entry
            if(present(sparse_x)) feature=sparse_x%indices(entry)
            z=work(feature)
            if(wscale*z>0) then
              work(feature)=max(0._dp,z-(u+q(feature))/wscale)
            else if(wscale*z<0) then
              work(feature)=min(0._dp,z+(u-q(feature))/wscale)
            end if
            q(feature)=q(feature)+wscale*(work(feature)-z)
          end do
        end if
        self%t_=self%t_+1
      end do
      self%standard_coef=work*wscale
      call require(all(ieee_is_finite(self%standard_coef)).and.ieee_is_finite(self%standard_intercept), &
        'SGD fit overflow; scale the inputs')
      if(self%early_stopping)then
        previous_stale=stale
        ! Source passes the backing coefficient buffer (without lazy wscale)
        ! to its validation callback, and never substitutes averaged weights.
        do j=1,size(validation_indices)
          i=validation_indices(j);inner=0
          if(present(sparse_x))then
            do k=sparse_x%indptr(i),sparse_x%indptr(i+1)-1
              inner=inner+work(sparse_x%indices(k))*sparse_x%data(k)
            end do
          else
            inner=dot_product(x(i,:),work)
          end if
          validation_prediction(j)=inner+self%standard_intercept
        end do
        if(self%classification)then
          if(self%loss=='log_loss')then
            ! sklearn 1.6.1 encodes binomial targets as 0/1 but installs
            ! callback classes [-1,1]: negative targets never score correct.
            validation_score=sum(score_weight*merge(1._dp,0._dp, &
              validation_prediction>0.and.y(validation_indices)>0))/sum(score_weight)
          else
            validation_score=sum(score_weight*merge(1._dp,0._dp, &
              merge(1._dp,-1._dp,validation_prediction>0)==y(validation_indices)))/sum(score_weight)
          end if
        else
          validation_score=r2_score(y(validation_indices),validation_prediction,sample_weight=score_weight)
        end if
        stale=0
        if(.not.ieee_is_nan(validation_score))then
          if(stopping.and.self%tol>=0)then
            if(validation_score<best_score+self%tol)stale=previous_stale+1
          end if
          if(validation_score>best_score)best_score=validation_score
        end if
      else if(stopping.and.self%tol>=0)then
        if(sumloss>best_loss-self%tol*n) then
          stale=stale+1
        else
          stale=0
        end if
        best_loss=min(best_loss,sumloss)
      end if
      if(stopping.and.self%tol>=0) then
        if(stale>=self%n_iter_no_change) then
          if(self%learning_rate=='adaptive'.and.eta>1.e-6_dp) then
            eta=eta/5;stale=0
          else
            self%converged_=.true.;exit
          end if
        end if
      end if
    end do
    self%n_iter_=min(it,epochs); self%loss_=sumloss/n
    call reset_scale();self%standard_coef=work
    self%t_=start_t+real(self%n_iter_,dp)*n
    self%coef_=self%standard_coef; self%intercept_=self%standard_intercept
    if(self%average>0.and.self%t_-1>=self%average) then
      self%coef_=self%average_coef; self%intercept_=self%average_intercept
    end if
  contains
    subroutine reset_scale()
      if(self%average>0) then
        self%average_coef=(self%average_coef+average_a*work)/average_b;average_a=0;average_b=1
      end if
      work=work*wscale;wscale=1
    end subroutine
  end subroutine
  integer function next_random(state) result(value)
    integer(int64), intent(inout) :: state
    if(state==0) state=1
    state=iand(ieor(state,ishft(state,13)),4294967295_int64)
    state=ieor(state,ishft(state,-17))
    state=iand(ieor(state,ishft(state,5)),4294967295_int64)
    value=int(iand(state,2147483647_int64))
  end function
  function sgd_predict(self,x) result(y)
    class(sgd_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call require(allocated(self%coef_),'SGDRegressor is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SGD feature count mismatch')
    y=matmul(x,self%coef_)+self%intercept_
  end function
  real(dp) function sgd_score(self,x,y) result(value)
    class(sgd_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
  subroutine sgdc_configure(self,head)
    class(sgd_classifier), intent(in) :: self
    type(sgd_regressor), intent(inout) :: head
    head%classification=.true.; head%loss=self%loss; head%penalty=self%penalty; head%learning_rate=self%learning_rate
    head%alpha=self%alpha; head%eta0=self%eta0; head%power_t=self%power_t; head%tol=self%tol
    head%l1_ratio=self%l1_ratio
    head%epsilon=self%epsilon
    head%early_stopping=self%early_stopping;head%validation_fraction=self%validation_fraction
    head%max_iter=self%max_iter; head%n_iter_no_change=self%n_iter_no_change; head%average=self%average
    head%random_state=self%random_state; head%fit_intercept=self%fit_intercept; head%shuffle=self%shuffle
    call sgd_controls(head)
  end subroutine
  subroutine sgdc_fit(self,x,y,sample_weight,class_weight,coef_init,intercept_init,sparse_x)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    real(dp), intent(in), optional :: coef_init(:,:),intercept_init(:)
    type(csr_matrix), optional, intent(in) :: sparse_x
    real(dp), allocatable :: initial(:,:),bias(:)
    if(present(coef_init)) then
      initial=coef_init
    else if(self%warm_start.and.allocated(self%coef_)) then
      initial=self%coef_
    end if
    if(present(intercept_init)) then
      bias=intercept_init
    else if(self%warm_start.and.allocated(self%intercept_)) then
      bias=self%intercept_
    end if
    call sgdc_train(self,x,y,.false.,sample_weight,class_weight,coef_init=initial,intercept_init=bias,sparse_x=sparse_x)
  end subroutine
  subroutine sgdc_partial(self,x,y,classes,sample_weight,class_weight)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    call sgdc_train(self,x,y,.true.,sample_weight,class_weight,classes)
  end subroutine
  subroutine sgdc_fit_vector(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),coef_init(:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:),intercept_init(:)
    call sgdc_fit(self,x,y,sample_weight,class_weight,reshape(coef_init,[1,size(coef_init)]),intercept_init)
  end subroutine
  subroutine sgdc_fit_scalar_bias(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),intercept_init
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:),coef_init(:,:)
    call sgdc_fit(self,x,y,sample_weight,class_weight,coef_init,[intercept_init])
  end subroutine
  subroutine sgdc_fit_vector_scalar_bias(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),coef_init(:),intercept_init
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:)
    call sgdc_fit(self,x,y,sample_weight,class_weight,reshape(coef_init,[1,size(coef_init)]),[intercept_init])
  end subroutine
  subroutine sgdc_train(self,x,y,incremental,sample_weight,class_weight,classes,coef_init,intercept_init,sparse_x)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    logical, intent(in) :: incremental
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: coef_init(:,:),intercept_init(:)
    type(csr_matrix), optional, intent(in) :: sparse_x
    type(sgd_regressor) :: settings
    type(legacy_random_state) :: rng
    integer, allocatable :: labels(:),encoded(:),seeds(:)
    real(dp), allocatable :: weights(:),cw(:),target(:),cost(:)
    integer :: n,p,k,channels,c,positive,i
    real(dp) :: start_t
    logical :: fresh
    logical,allocatable::validation_mask(:)
    call sgdc_configure(self,settings);n=size(y)
    call require(.not.(incremental.and.self%early_stopping),'early_stopping must be false with partial_fit')
    if(present(sparse_x)) then
      call check_csr(sparse_x,n);p=sparse_x%n_cols
    else
      call check_xy(x,n);p=size(x,2)
    end if
    fresh=.not.incremental
    if(incremental) fresh=.not.allocated(self%heads)
    if(fresh) then
      if(incremental) then
        call require(present(classes),'SGDClassifier first partial_fit requires classes')
        labels=unique_labels(classes)
        call require(size(labels)==size(classes),'SGDClassifier classes must be distinct')
      else
        labels=unique_labels(y)
      end if
      call require(size(labels)>=2,'SGDClassifier requires at least two classes')
    else
      labels=self%classes_
      call require(p==self%n_features_in_,'SGDClassifier feature count mismatch')
      call require(self%average==self%heads(1)%fitted_average,'SGD averaging cannot change during partial_fit')
      if(present(classes)) then
        encoded=unique_labels(classes)
        call require(size(encoded)==size(labels),'SGDClassifier classes changed')
        call require(all(encoded==labels),'SGDClassifier classes changed')
      end if
    end if
    k=size(labels)
    call require(.not.(incremental.and.self%average>0.and.self%fit_intercept), &
      'averaged SGDClassifier partial_fit requires fit_intercept=false')
    encoded=[(label_index(labels,y(i)),i=1,n)]
    call require(all(encoded>0),'SGDClassifier target is outside known classes')
    weights=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'SGD sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite SGD sample weights')
      call require(all(sample_weight>=0),'negative SGD sample weights'); weights=sample_weight
    end if
    validation_mask=spread(.false.,1,n)
    if(self%early_stopping)validation_mask=make_validation_mask(n,self%validation_fraction,self%random_state,weights,encoded)
    self%validation_mask_=validation_mask
    cw=spread(1._dp,1,k)
    call require(.not.(incremental.and.self%class_weight_balanced),'balanced SGD class weights require fit')
    if(present(class_weight)) then
      call require(.not.self%class_weight_balanced,'choose balanced or explicit SGD class weights')
      call require(size(class_weight)==k,'SGD class weight count mismatch')
      call require(all(ieee_is_finite(class_weight)),'nonfinite SGD class weights')
      call require(all(class_weight>0),'SGD class weights must be positive'); cw=class_weight
    else if(self%class_weight_balanced) then
      do c=1,k
        cw(c)=real(n,dp)/(k*real(count(encoded==c),dp))
      end do
    end if
    channels=k
    if(k==2) channels=1
    if(present(coef_init)) then
      if(channels==1) then
        call require(size(coef_init)==p,'SGDClassifier coefficient initializer shape mismatch')
      else
        call require(all(shape(coef_init)==[channels,p]),'SGDClassifier coefficient initializer shape mismatch')
      end if
      call require(all(ieee_is_finite(coef_init)),'SGD coefficient initializer must be finite')
    end if
    if(present(intercept_init)) then
      call require(size(intercept_init)==channels,'SGDClassifier intercept initializer shape mismatch')
      call require(all(ieee_is_finite(intercept_init)),'SGD intercept initializer must be finite')
    end if
    allocate(seeds(channels));seeds=self%random_state
    if(k>2) then
      call rng%seed(self%random_state)
      do c=1,channels
        seeds(c)=rng%integer_index(2147483647)-1
      end do
    end if
    if(fresh) then
      if(allocated(self%heads)) deallocate(self%heads)
      allocate(self%heads(channels)); self%classes_=labels; self%n_features_in_=p; self%t_=1
      do c=1,channels
        call sgdc_configure(self,self%heads(c)); call sgd_initialize(self%heads(c),p)
        if(present(coef_init)) then
          if(channels==1) then
            self%heads(c)%standard_coef=reshape(transpose(coef_init),[p])
          else
            self%heads(c)%standard_coef=coef_init(c,:)
          end if
        end if
        if(present(intercept_init)) self%heads(c)%standard_intercept=intercept_init(c)
      end do
    end if
    self%coef_=spread(spread(0._dp,1,channels),2,p)
    self%intercept_=spread(0._dp,1,channels); self%loss_=self%intercept_
    self%n_iter_per_class_=spread(0,1,channels); self%converged_per_class_=spread(.false.,1,channels)
    start_t=self%t_
    do c=1,channels
      positive=c
      if(k==2) positive=2
      target=merge(1._dp,-1._dp,encoded==positive)
      if(k==2) then
        cost=weights*cw(encoded)
      else
        cost=weights*merge(cw(c),1._dp,encoded==c)
      end if
      call sgdc_configure(self,self%heads(c)); self%heads(c)%t_=start_t
      self%heads(c)%random_state=seeds(c)
      if(incremental) then
        call sgd_run(self%heads(c),x,target,1,.false.,cost,sparse_x)
      else
        call sgd_run(self%heads(c),x,target,self%max_iter,.true.,cost,sparse_x,validation_mask,weights)
      end if
      self%n_iter_per_class_(c)=self%heads(c)%n_iter_; self%converged_per_class_(c)=self%heads(c)%converged_
      self%loss_(c)=self%heads(c)%loss_
    end do
    self%n_iter_=maxval(self%n_iter_per_class_); self%t_=start_t+real(self%n_iter_,dp)*n
    self%converged_=all(self%converged_per_class_)
    do c=1,channels
      if(self%average>0.and.self%t_-1>=self%average) then
        self%coef_(c,:)=self%heads(c)%average_coef; self%intercept_(c)=self%heads(c)%average_intercept
      else
        self%coef_(c,:)=self%heads(c)%standard_coef; self%intercept_(c)=self%heads(c)%standard_intercept
      end if
    end do
  end subroutine
  function sgdc_decision(self,x) result(scores)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    call require(allocated(self%heads),'SGDClassifier is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SGDClassifier feature count mismatch')
    scores=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function sgdc_log_proba(self,x) result(values)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: i,j
    values=self%predict_proba(x)
    do j=1,size(values,2)
      do i=1,size(values,1)
        if(values(i,j)>0) then
          values(i,j)=log(values(i,j))
        else
          values(i,j)=ieee_value(0._dp,ieee_negative_inf)
        end if
      end do
    end do
  end function
  function sgdc_predict(self,x) result(y)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: scores(:,:)
    scores=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=self%classes_(merge(2,1,scores(:,1)>0))
    else
      y=self%classes_(maxloc(scores,dim=2))
    end if
  end function
  function sgdc_proba(self,x) result(prob)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),scores(:,:)
    scores=self%decision_function(x);prob=probabilities_from_scores(self,scores)
  end function
  function probabilities_from_scores(self,scores) result(prob)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: scores(:,:)
    real(dp), allocatable :: prob(:,:)
    real(dp) :: total
    integer :: i,c,k
    k=size(self%classes_)
    call require(self%loss=='log_loss'.or.self%loss=='modified_huber','SGD loss does not provide probabilities')
    allocate(prob(size(scores,1),k))
    do i=1,size(scores,1)
      do c=1,size(scores,2)
        if(self%loss=='log_loss') then
          prob(i,c)=sigmoid(scores(i,c))
        else
          prob(i,c)=(min(1._dp,max(-1._dp,scores(i,c)))+1)/2
        end if
      end do
      if(k==2) then
        prob(i,2)=prob(i,1); prob(i,1)=1-prob(i,2)
      else
        total=sum(prob(i,:))
        if(total>0) then
          prob(i,:)=prob(i,:)/total
        else
          prob(i,:)=1._dp/k
        end if
      end if
    end do
  end function
  real(dp) function sgdc_score(self,x,y) result(value)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function
  function sgd_predict_sparse(self,x) result(y)
    class(sgd_regressor), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: y(:),matrix(:,:)
    call require(allocated(self%coef_),'SGDRegressor is not fitted');call check_csr(x)
    call require(x%n_cols==self%n_features_in_,'SGD feature count mismatch')
    matrix=x%matmul(reshape(self%coef_,[size(self%coef_),1]));y=matrix(:,1)+self%intercept_
  end function
  real(dp) function sgd_score_sparse(self,x,y) result(value)
    class(sgd_regressor), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:)
    value=r2_score(y,self%predict(x))
  end function
  function sgdc_decision_sparse(self,x) result(scores)
    class(sgd_classifier), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: scores(:,:)
    call require(allocated(self%heads),'SGDClassifier is not fitted');call check_csr(x)
    call require(x%n_cols==self%n_features_in_,'SGDClassifier feature count mismatch')
    scores=x%matmul(transpose(self%coef_))+spread(self%intercept_,1,x%n_rows)
  end function
  function sgdc_predict_sparse(self,x) result(y)
    class(sgd_classifier), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    integer, allocatable :: y(:)
    real(dp), allocatable :: scores(:,:)
    scores=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=self%classes_(merge(2,1,scores(:,1)>0))
    else
      y=self%classes_(maxloc(scores,dim=2))
    end if
  end function
  function sgdc_proba_sparse(self,x) result(prob)
    class(sgd_classifier), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: prob(:,:),scores(:,:)
    scores=self%decision_function(x);prob=probabilities_from_scores(self,scores)
  end function
  function sgdc_log_proba_sparse(self,x) result(values)
    class(sgd_classifier), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: values(:,:)
    integer :: i,j
    values=self%predict_proba(x)
    do j=1,size(values,2)
      do i=1,size(values,1)
        if(values(i,j)>0) then
          values(i,j)=log(values(i,j))
        else
          values(i,j)=ieee_value(0._dp,ieee_negative_inf)
        end if
      end do
    end do
  end function
  real(dp) function sgdc_score_sparse(self,x,y) result(value)
    class(sgd_classifier), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function
  subroutine sgdc_partial_sparse(self,x,y,classes,sample_weight,class_weight)
    class(sgd_classifier), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: y(:)
    integer, optional, intent(in) :: classes(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:)
    real(dp) :: empty(0,0)
    call sgdc_train(self,empty,y,.true.,sample_weight,class_weight,classes,sparse_x=x)
  end subroutine
  subroutine sgdc_fit_sparse(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(sgd_classifier), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:),coef_init(:,:),intercept_init(:)
    real(dp) :: empty(0,0)
    call sgdc_fit(self,empty,y,sample_weight,class_weight,coef_init,intercept_init,x)
  end subroutine
  subroutine sgdc_fit_sparse_vector(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(sgd_classifier), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: coef_init(:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:),intercept_init(:)
    call sgdc_fit_sparse(self,x,y,sample_weight,class_weight,reshape(coef_init,[1,size(coef_init)]),intercept_init)
  end subroutine
  subroutine sgdc_fit_sparse_scalar(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(sgd_classifier), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: intercept_init
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:),coef_init(:,:)
    call sgdc_fit_sparse(self,x,y,sample_weight,class_weight,coef_init,[intercept_init])
  end subroutine
  subroutine sgdc_fit_sparse_vector_scalar(self,x,y,sample_weight,class_weight,coef_init,intercept_init)
    class(sgd_classifier), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: coef_init(:),intercept_init
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:),class_weight(:)
    call sgdc_fit_sparse(self,x,y,sample_weight,class_weight,reshape(coef_init,[1,size(coef_init)]),[intercept_init])
  end subroutine
end module
