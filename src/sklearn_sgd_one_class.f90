! Dense/CSR SGDOneClassSVM updates follow sklearn 1.6.1 _stochastic_gradient.py,
! _sgd_fast.pyx.tp, _weight_vector.pyx.tp and _seq_dataset.pyx.tp (BSD notices retained).
module sklearn_sgd_one_class
  use sklearn_common, only: dp,require,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse,only:csr_matrix
  use iso_fortran_env, only: int64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sgd_one_class_svm
  type :: sgd_one_class_svm
    real(dp) :: nu=.5_dp,eta0=0,power_t=.5_dp,tol=1.e-3_dp,t_=1,offset_=0,loss_=0
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0,n_features_in_=0
    character(16) :: learning_rate='optimal'
    logical :: fit_intercept=.true.,shuffle=.true.,warm_start=.false.,converged_=.false.
    real(dp), allocatable :: coef_(:)
    real(dp), allocatable, private :: standard_coef(:),average_coef(:)
    real(dp), private :: standard_intercept=1,average_intercept=0
    integer, private :: fitted_average=0
  contains
    procedure,private::fit_dense=>fit,fit_sparse
    generic::fit=>fit_dense,fit_sparse
    procedure,private::partial_dense=>partial_fit,partial_sparse
    generic::partial_fit=>partial_dense,partial_sparse
    procedure,private::decision_dense=>decision_function,decision_sparse
    generic::decision_function=>decision_dense,decision_sparse
    procedure,private::samples_dense=>score_samples,samples_sparse
    generic::score_samples=>samples_dense,samples_sparse
    procedure,private::predict_dense=>predict,predict_sparse
    generic::predict=>predict_dense,predict_sparse
    procedure,private::fit_predict_dense=>fit_predict,fit_predict_sparse
    generic::fit_predict=>fit_predict_dense,fit_predict_sparse
  end type
contains
  subroutine controls(self)
    class(sgd_one_class_svm), intent(in) :: self
    call require(all(ieee_is_finite([self%nu,self%eta0,self%power_t,self%tol])),'nonfinite SGDOneClassSVM controls')
    call require(self%nu>0.and.self%nu<=1,'SGDOneClassSVM nu must be in (0,1]')
    call require(self%nu/2>0,'SGDOneClassSVM nu is too small')
    call require(any(self%learning_rate==[character(16)::'constant','optimal','invscaling','adaptive']), &
      'unknown SGDOneClassSVM learning rate')
    call require(self%eta0>=0,'SGDOneClassSVM eta0 must be nonnegative')
    if(self%learning_rate/='optimal') call require(self%eta0>0,'SGDOneClassSVM eta0 must be positive for this schedule')
    call require(self%tol>=0.or.self%tol==-1,'SGDOneClassSVM tol must be nonnegative or -1')
    call require(self%max_iter>0.and.self%n_iter_no_change>0,'invalid SGDOneClassSVM iteration controls')
    call require(self%average>=0,'SGDOneClassSVM average must be nonnegative')
    call require(self%random_state>=0,'SGDOneClassSVM seed must be nonnegative')
  end subroutine
  subroutine initialize(self,p,coef,offset)
    class(sgd_one_class_svm), intent(inout) :: self
    integer, intent(in) :: p
    real(dp), intent(in) :: coef(:),offset
    call require(size(coef)==p,'SGDOneClassSVM coefficient initializer shape mismatch')
    call require(all(ieee_is_finite(coef)).and.ieee_is_finite(offset),'nonfinite SGDOneClassSVM initializer')
    self%standard_coef=coef; self%average_coef=spread(0._dp,1,p)
    self%standard_intercept=1-offset; self%average_intercept=0
    self%coef_=coef; self%offset_=offset; self%t_=1; self%n_features_in_=p; self%fitted_average=self%average
  end subroutine
  subroutine fit(self,x,sample_weight,coef_init,offset_init)
    class(sgd_one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:),coef_init(:),offset_init
    real(dp), allocatable :: initial(:)
    real(dp) :: offset
    call controls(self); call check_x(x)
    initial=spread(0._dp,1,size(x,2)); offset=0
    if(self%warm_start.and.allocated(self%coef_)) then
      initial=self%coef_; offset=self%offset_
    end if
    if(present(coef_init)) initial=coef_init
    if(present(offset_init)) offset=offset_init
    call initialize(self,size(x,2),initial,offset)
    call run(self,x,self%max_iter,.true.,sample_weight)
  end subroutine
  subroutine partial_fit(self,x,sample_weight)
    class(sgd_one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call controls(self); call check_x(x)
    if(.not.allocated(self%standard_coef)) call initialize(self,size(x,2),spread(0._dp,1,size(x,2)),0._dp)
    call require(size(x,2)==self%n_features_in_,'SGDOneClassSVM feature count mismatch')
    call require(self%average==self%fitted_average,'SGDOneClassSVM averaging cannot change during partial_fit')
    call run(self,x,1,.false.,sample_weight)
  end subroutine
  integer function next_random(state) result(value)
    integer(int64), intent(inout) :: state
    if(state==0) state=1
    state=iand(ieor(state,ishft(state,13)),4294967295_int64)
    state=ieor(state,ishft(state,-17))
    state=iand(ieor(state,ishft(state,5)),4294967295_int64)
    value=int(iand(state,2147483647_int64))
  end function
  subroutine check_csr(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'X must be nonempty')
    call require(all(ieee_is_finite(x%data)),'X contains NaN or infinity')
  end subroutine
  subroutine fit_sparse(self,x,sample_weight,coef_init,offset_init)
    class(sgd_one_class_svm),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:),coef_init(:),offset_init
    real(dp),allocatable::initial(:)
    real(dp)::offset
    call controls(self);call check_csr(x)
    initial=spread(0._dp,1,x%n_cols);offset=0
    if(self%warm_start.and.allocated(self%coef_))then
      initial=self%coef_;offset=self%offset_
    end if
    if(present(coef_init))initial=coef_init
    if(present(offset_init))offset=offset_init
    call initialize(self,x%n_cols,initial,offset)
    call run(self,epochs=self%max_iter,stopping=.true.,sample_weight=sample_weight,sparse=x)
  end subroutine
  subroutine partial_sparse(self,x,sample_weight)
    class(sgd_one_class_svm),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    call controls(self);call check_csr(x)
    if(.not.allocated(self%standard_coef))call initialize(self,x%n_cols,spread(0._dp,1,x%n_cols),0._dp)
    call require(x%n_cols==self%n_features_in_,'SGDOneClassSVM feature count mismatch')
    call require(self%average==self%fitted_average,'SGDOneClassSVM averaging cannot change during partial_fit')
    call run(self,epochs=1,stopping=.false.,sample_weight=sample_weight,sparse=x)
  end subroutine
  subroutine run(self,x,epochs,stopping,sample_weight,sparse)
    class(sgd_one_class_svm), intent(inout) :: self
    real(dp),optional, intent(in) :: x(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    integer, intent(in) :: epochs
    logical, intent(in) :: stopping
    real(dp), optional, intent(in) :: sample_weight(:)
    type(legacy_random_state) :: rng
    real(dp),allocatable::w(:),aw(:),weights(:)
    integer,allocatable::order(:)
    real(dp) :: b,ab,alpha,eta,optimal_init,typw,intercept_decay
    real(dp) :: prediction,update,wscale,average_a,average_b,count,mu,sumloss,best_loss,factor
    integer :: n,p,i,j,k,it,stale,pick,tmp,seed,column
    integer(int64) :: state
    intercept_decay=1
    if(present(sparse))then
      n=sparse%n_rows;p=sparse%n_cols;intercept_decay=.01_dp
    else
      n=size(x,1);p=size(x,2)
    end if
    allocate(w(p),aw(p),weights(n),order(n));weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'SGDOneClassSVM sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'SGDOneClassSVM weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'SGDOneClassSVM weights must be finite and nonnegative')
      weights=sample_weight
    end if
    call rng%seed(self%random_state); seed=rng%integer_index(2147483647)-1
    w=self%standard_coef; aw=self%average_coef; b=self%standard_intercept; ab=self%average_intercept
    wscale=1; average_a=0; average_b=1; alpha=self%nu/2; eta=self%eta0; optimal_init=0
    if(self%learning_rate=='optimal') then
      typw=sqrt(1/sqrt(alpha)); optimal_init=1/(typw*alpha)
    end if
    order=[(i,i=1,n)]; best_loss=huge(1._dp); stale=0; self%converged_=.false.
    do it=1,epochs
      if(self%shuffle) then
        ! Dataset.shuffle receives the same seed by value each epoch; its
        ! existing index permutation is shuffled in place by a forward pass.
        state=int(seed,int64)
        do j=1,n-1
          pick=j+modulo(next_random(state),n-j+1); tmp=order(j); order(j)=order(pick); order(pick)=tmp
        end do
      end if
      sumloss=0
      do j=1,n
        i=order(j); prediction=0
        if(present(sparse))then
          do k=sparse%indptr(i),sparse%indptr(i+1)-1
            prediction=prediction+w(sparse%indices(k))*sparse%data(k)
          end do
        else
          do k=1,p
            prediction=prediction+w(k)*x(i,k)
          end do
        end if
        prediction=prediction*wscale+b
        select case(trim(self%learning_rate))
        case('optimal'); eta=1/(alpha*(optimal_init+self%t_-1))
        case('invscaling'); eta=self%eta0/self%t_**self%power_t
        end select
        sumloss=sumloss+max(0._dp,1-prediction)
        update=0
        if(prediction<=1) update=eta*weights(i)
        factor=max(0._dp,1-eta*alpha); wscale=wscale*factor
        if(wscale<1.e-9_dp) call reset_scale()
        if(update/=0) then
          if(present(sparse))then
            do k=sparse%indptr(i),sparse%indptr(i+1)-1
              column=sparse%indices(k);w(column)=w(column)+sparse%data(k)*(update/wscale)
            end do
          else
            do k=1,p
              w(k)=w(k)+x(i,k)*(update/wscale)
            end do
          end if
        end if
        if(self%fit_intercept) b=b+(update-2*eta*alpha)*intercept_decay
        if(self%average>0.and.self%t_>=self%average) then
          count=self%t_-self%average+1; mu=1/count
          if(present(sparse))then
            do k=sparse%indptr(i),sparse%indptr(i+1)-1
              column=sparse%indices(k);aw(column)=aw(column)+average_a*sparse%data(k)*(-update/wscale)
            end do
          else
            do k=1,p
              aw(k)=aw(k)+average_a*x(i,k)*(-update/wscale)
            end do
          end if
          if(count>1) average_b=average_b/(1-mu)
          average_a=average_a+mu*average_b*wscale
          ab=ab+(b-ab)/count
        end if
        self%t_=self%t_+1
      end do
      call require(all(ieee_is_finite(w)).and.ieee_is_finite(b),'SGDOneClassSVM fit overflow; scale the inputs')
      if(stopping.and.self%tol>=0) then
        if(sumloss>best_loss-self%tol*n) then
          stale=stale+1
        else
          stale=0
        end if
        best_loss=min(best_loss,sumloss)
        if(stale>=self%n_iter_no_change) then
          if(self%learning_rate=='adaptive'.and.eta>1.e-6_dp) then
            eta=eta/5; stale=0
          else
            self%converged_=.true.
            exit
          end if
        end if
      end if
    end do
    self%n_iter_=min(it,epochs); self%loss_=sumloss/n
    call reset_scale()
    self%standard_coef=w; self%standard_intercept=b; self%average_coef=aw; self%average_intercept=ab
    self%coef_=w; self%offset_=1-b
    if(self%average>0.and.self%t_-1>=self%average) then
      self%coef_=aw; self%offset_=1-ab
    end if
  contains
    subroutine reset_scale()
      if(self%average>0) then
        aw=(aw+average_a*w)/average_b; average_a=0; average_b=1
      end if
      w=w*wscale; wscale=1
    end subroutine
  end subroutine
  function score_samples(self,x) result(scores)
    class(sgd_one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1))
    call require(allocated(self%coef_),'SGDOneClassSVM is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SGDOneClassSVM feature count mismatch')
    scores=matmul(x,self%coef_)
  end function
  function decision_function(self,x) result(scores)
    class(sgd_one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1))
    scores=self%score_samples(x)-self%offset_
  end function
  function predict(self,x) result(labels)
    class(sgd_one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    labels=merge(1,-1,self%decision_function(x)>=0)
  end function
  function fit_predict(self,x,sample_weight) result(labels)
    class(sgd_one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer :: labels(size(x,1))
    call self%fit(x,sample_weight); labels=self%predict(x)
  end function
  function samples_sparse(self,x)result(scores)
    class(sgd_one_class_svm),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::scores(:)
    integer::i,k
    call require(allocated(self%coef_),'SGDOneClassSVM is not fitted');call check_csr(x)
    call require(x%n_cols==self%n_features_in_,'SGDOneClassSVM feature count mismatch')
    allocate(scores(x%n_rows));scores=0
    do i=1,x%n_rows
      do k=x%indptr(i),x%indptr(i+1)-1
        scores(i)=scores(i)+self%coef_(x%indices(k))*x%data(k)
      end do
    end do
  end function
  function decision_sparse(self,x)result(scores)
    class(sgd_one_class_svm),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::scores(:)
    scores=self%score_samples(x)-self%offset_
  end function
  function predict_sparse(self,x)result(labels)
    class(sgd_one_class_svm),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::labels(:)
    labels=merge(1,-1,self%decision_function(x)>=0)
  end function
  function fit_predict_sparse(self,x,sample_weight)result(labels)
    class(sgd_one_class_svm),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    integer,allocatable::labels(:)
    call self%fit(x,sample_weight);labels=self%predict(x)
  end function
end module
