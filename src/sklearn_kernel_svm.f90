! Dense/CSR SVR, NuSVR and one-class SVM contracts from sklearn/svm/_base.py,
! _classes.py and its bundled LIBSVM. SVR/OneClassSVM use translated Solver; NuSVR uses Solver_NU.
! See THIRD_PARTY_NOTICES.md and LICENSES/libsvm-COPYRIGHT for attribution.
module sklearn_kernel_svm
  use sklearn_common
  use sklearn_metrics, only: rbf_kernel,linear_kernel,polynomial_kernel,sigmoid_kernel,r2_score
  use sklearn_libsvm_solver,only:libsvm_solve,libsvm_solve_nu
  use,intrinsic::iso_fortran_env,only:real32
  use sklearn_sparse, only:csr_matrix,csr_from_dense
  use sklearn_stateless_preprocessing, only:sorted_sparse
  use sklearn_libsvm_sparse,only:sparse_inner,svm_sorted,sparse_rows,libsvm_sparse_gram
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_fma
  implicit none
  private
  public :: svr,nu_svr,one_class_svm
  type :: kernel_model
    character(16) :: kernel='rbf',gamma_rule='scale'
    real(dp) :: gamma=-1,gamma_=0,coef0=0,tol=1.e-3_dp,intercept_=0,kkt_violation_=0
    integer :: degree=3,max_iter=100000,n_iter_=0,fit_status_=1,n_support_=0
    logical :: converged_=.false.
    logical :: sparse_fit_=.false.
    type(csr_matrix), allocatable :: x_fit_csr_,support_vectors_csr_
    integer, allocatable :: support_(:)
    real(dp), allocatable :: support_vectors_(:,:),dual_coef_(:),coef_(:),x_fit_(:,:)
    real(dp), allocatable, private :: dual_full_(:)
    character(16), private :: kernel_fit_='rbf'
    integer, private :: degree_fit_=3
    real(dp), private :: coef0_fit_=0
  contains
    procedure,private :: decision_dense=>kernel_decision,decision_csr=>kernel_decision_csr
    generic :: decision_function=>decision_dense,decision_csr
  end type
  type, extends(kernel_model) :: regression_model
    real(dp) :: c=1
  contains
    procedure,private :: predict_dense=>regression_predict,predict_csr=>regression_predict_csr
    generic :: predict=>predict_dense,predict_csr
    procedure,private :: score_dense=>regression_score,score_csr=>regression_score_csr
    generic :: score=>score_dense,score_csr
  end type
  type, extends(regression_model) :: svr
    real(dp) :: epsilon=0.1_dp
    logical::shrinking=.true.
  contains
    procedure,private :: fit_dense=>svr_fit,fit_csr=>svr_fit_csr
    generic :: fit=>fit_dense,fit_csr
  end type
  type, extends(regression_model) :: nu_svr
    real(dp) :: nu=0.5_dp,epsilon_=0,epsilon_signed_=0
    logical::shrinking=.true.
    real(dp),allocatable::dual_variables_(:,:)
  contains
    procedure,private :: fit_dense=>nusvr_fit,fit_csr=>nusvr_fit_csr
    generic :: fit=>fit_dense,fit_csr
  end type
  type, extends(kernel_model) :: one_class_svm
    real(dp) :: nu=0.5_dp,offset_=0
    logical::shrinking=.true.
  contains
    procedure,private :: fit_dense=>one_fit,fit_csr=>one_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure,private :: predict_dense=>one_predict,predict_csr=>one_predict_csr
    generic :: predict=>predict_dense,predict_csr
    procedure,private :: fit_predict_dense=>one_fit_predict,fit_predict_csr=>one_fit_predict_csr
    generic :: fit_predict=>fit_predict_dense,fit_predict_csr
    procedure,private :: score_samples_dense=>one_score_samples,score_samples_csr=>one_score_samples_csr
    generic :: score_samples=>score_samples_dense,score_samples_csr
  end type
contains
  function gram_matrix(self,x,y) result(k)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(self%kernel_fit_)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('linear'); k=linear_kernel(x,y)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree_fit_,self%gamma_,self%coef0_fit_)
    case('sigmoid');k=sigmoid_kernel(x,y,self%gamma_,self%coef0_fit_)
    case('precomputed'); k=x
    end select
  end function
  subroutine setup_controls(self)
    class(kernel_model), intent(inout) :: self
    call require(self%max_iter>0,'kernel SVM max_iter must be positive')
    call require(ieee_is_finite(self%tol),'invalid kernel SVM tolerance')
    call require(self%tol>0,'invalid kernel SVM tolerance')
    call require(ieee_is_finite(self%gamma).and.ieee_is_finite(self%coef0),'invalid kernel SVM parameters')
    call require((self%gamma>=0.or.self%gamma==-1).and.self%degree>=0,'invalid kernel SVM parameters')
    call require(any(self%kernel==[character(16)::'rbf','linear','poly','polynomial','sigmoid','precomputed']), &
      'unsupported kernel SVM kernel')
    call require(self%gamma_rule=='scale'.or.self%gamma_rule=='auto','invalid kernel SVM gamma rule')
    self%kernel_fit_=self%kernel; self%degree_fit_=self%degree; self%coef0_fit_=self%coef0
    if(allocated(self%coef_)) deallocate(self%coef_)
    if(allocated(self%support_vectors_csr_))deallocate(self%support_vectors_csr_)
  end subroutine
  subroutine setup(self,x,gram)
    class(kernel_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: gram(:,:)
    real(dp) :: variance,mean,diagonal
    integer :: n,i,j
    call check_x(x); n=size(x,1);call setup_controls(self)
    self%sparse_fit_=.false.
    if(allocated(self%x_fit_csr_))deallocate(self%x_fit_csr_)
    self%x_fit_=x; self%gamma_=self%gamma
    if(self%gamma_==-1) then
      self%gamma_=1._dp/size(x,2)
      if(self%gamma_rule=='scale') then
        mean=sum(x)/size(x); variance=sum((x-mean)**2)/size(x); self%gamma_=1
        if(variance>0) self%gamma_=1/(size(x,2)*variance)
      end if
    end if
    if(self%kernel_fit_=='precomputed') then
      self%gamma_=0
      call require(size(x,2)==n,'kernel SVM precomputed matrix must be square')
    end if
    gram=gram_matrix(self,x,x)
    ! The installed SciPy BLAS scalar ddot path (width < 16) uses fused
    ! accumulation. LIBSVM keeps these diagonals in double precision.
    if(self%kernel_fit_/='precomputed'.and.size(x,2)<16)then
      do i=1,n
        diagonal=0
        do j=1,size(x,2)
          diagonal=ieee_fma(x(i,j),x(i,j),diagonal)
        end do
        select case(self%kernel_fit_)
        case('linear');gram(i,i)=diagonal
        case('poly','polynomial');gram(i,i)=(self%gamma_*diagonal+self%coef0_fit_)**self%degree_fit_
        case('sigmoid');gram(i,i)=tanh(self%gamma_*diagonal+self%coef0_fit_)
        end select
      end do
    end if
    call require(all(ieee_is_finite(gram)),'kernel SVM nonfinite Gram matrix')
  end subroutine
  subroutine check_csr(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'kernel SVM CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'kernel SVM CSR input must be finite')
  end subroutine
  subroutine setup_csr(self,x,gram)
    class(kernel_model),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable,intent(out)::gram(:,:)
    type(csr_matrix)::canonical
    real(dp)::mean,variance,mass
    call check_csr(x);call setup_controls(self)
    call require(self%kernel_fit_/='precomputed','kernel SVM precomputed inputs must be dense')
    self%x_fit_csr_=svm_sorted(x);self%sparse_fit_=.true.
    if(allocated(self%x_fit_))deallocate(self%x_fit_)
    if(allocated(self%support_vectors_))deallocate(self%support_vectors_)
    self%gamma_=self%gamma
    if(self%gamma_==-1)then
      self%gamma_=1._dp/x%n_cols
      if(self%gamma_rule=='scale')then
        ! SciPy multiply/mean coalesce duplicates for the second moment.
        canonical=sorted_sparse(x,.true.);mass=real(x%n_rows,dp)*real(x%n_cols,dp)
        mean=sum(canonical%data)/mass
        variance=sum(canonical%data**2)/mass-mean**2;self%gamma_=1
        if(variance/=0)self%gamma_=1/(x%n_cols*variance)
      end if
    end if
    gram=sparse_gram(self,self%x_fit_csr_,self%x_fit_csr_,.true.)
    call require(all(ieee_is_finite(gram)),'kernel SVM nonfinite Gram matrix')
  end subroutine
  function sparse_gram(self,x,y,training) result(k)
    class(kernel_model),intent(in)::self
    type(csr_matrix),intent(in)::x,y
    logical,intent(in)::training
    real(dp),allocatable::k(:,:)
    k=libsvm_sparse_gram(x,y,self%kernel_fit_,self%gamma_,self%coef0_fit_,self%degree_fit_,training)
  end function
  function bounds(n,c,sample_weight) result(cap)
    integer, intent(in) :: n
    real(dp), intent(in) :: c
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: cap(n)
    call require(ieee_is_finite(c),'kernel SVM C must be positive')
    call require(c>0,'kernel SVM C must be positive'); cap=c
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'kernel SVM weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'invalid kernel SVM sample weights')
      call require(all(sample_weight>=0).and.any(sample_weight>0),'invalid kernel SVM sample weights')
      cap=c*sample_weight
    end if
    call require(all(ieee_is_finite(cap)),'kernel SVM capacity overflow')
  end function
  subroutine select_pair(g,alpha,cap,signs,index,group,epsilon,groups,first,last,gap)
    real(dp), intent(in) :: g(:),alpha(:),cap(:),signs(:),epsilon
    integer, intent(in) :: index(:),group(:),groups
    integer, intent(out) :: first,last
    real(dp), intent(out) :: gap
    real(dp) :: low(groups),high(groups),value
    integer :: left(groups),right(groups),j,h
    low=huge(1._dp); high=-huge(1._dp); left=0; right=0
    do j=1,size(alpha)
      if(cap(j)==0) cycle
      h=group(j); value=g(index(j))+signs(j)*epsilon
      if((signs(j)>0.and.alpha(j)<cap(j)).or.(signs(j)<0.and.alpha(j)>0)) then
        if(value<low(h)) then
          low(h)=value; left(h)=j
        end if
      end if
      if((signs(j)>0.and.alpha(j)>0).or.(signs(j)<0.and.alpha(j)<cap(j))) then
        if(value>high(h)) then
          high(h)=value; right(h)=j
        end if
      end if
    end do
    gap=0; first=0; last=0
    do h=1,groups
      if(left(h)==0.or.right(h)==0) cycle
      if(high(h)-low(h)>gap) then
        gap=high(h)-low(h); first=left(h); last=right(h)
      end if
    end do
  end subroutine
  subroutine optimize(self,gram,y,capacity,mode,epsilon,nu,levels)
    class(kernel_model), intent(inout) :: self
    real(dp), intent(in) :: gram(:,:),y(:),capacity(:),epsilon,nu
    integer, intent(in) :: mode ! 1: epsilon SVR; 2: nu SVR; 3: one-class
    real(dp), intent(out) :: levels(2)
    real(dp), allocatable :: alpha(:),cap(:),signs(:),g(:),beta(:),qcache(:,:),qd(:),initial(:),ones(:),zeros(:),linear_term(:),solver_cap(:)
    integer, allocatable :: index(:),group(:),positive(:)
    real(dp) :: remaining,target,rho,nu_level
    integer :: n,m,groups,i,j,a,b,np
    n=size(y); m=2*n; groups=1
    if(mode==3) m=n
    if(mode==2) groups=2
    allocate(alpha(m),cap(m),signs(m),g(n),beta(n),index(m),group(m)); alpha=0; signs=1; group=1
    index=[(1+mod(i-1,n),i=1,m)]; cap=capacity(index)
    if(mode/=3) signs(n+1:)=-1
    if(mode==2) group(n+1:)=2
    if(mode==3)then
      positive=pack([(i,i=1,n)],capacity>0);np=size(positive)
      allocate(qcache(np,np),qd(np),initial(np),ones(np),zeros(np))
      target=0
      do i=1,np
        target=target+capacity(positive(i))*nu
      end do
      remaining=target;initial=0;ones=1;zeros=0
      do i=1,np
        if(remaining<=0)exit
        initial(i)=min(capacity(positive(i)),remaining);remaining=remaining-initial(i)
      end do
      do i=1,np
        qd(i)=gram(positive(i),positive(i))
        do j=1,np
          qcache(j,i)=real(real(gram(positive(i),positive(j)),real32),dp)
        end do
      end do
      select type(self)
      class is(one_class_svm)
        call libsvm_solve(qcache,qd,zeros,ones,capacity(positive),initial,self%tol,self%max_iter, &
          self%shrinking,rho,self%n_iter_,self%converged_)
      class default
        call require(.false.,'internal one-class solver type mismatch')
      end select
      alpha(positive)=initial;beta=alpha;g=matmul(gram,beta)
      call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,self%kkt_violation_)
      self%fit_status_=merge(0,1,self%converged_);levels=0;levels(1)=rho
    else if(mode==1.or.mode==2)then
      positive=pack([(i,i=1,n)],capacity>0);np=size(positive)
      allocate(qcache(2*np,2*np),qd(2*np),initial(2*np),ones(2*np),linear_term(2*np),solver_cap(2*np))
      initial=0;ones(:np)=1;ones(np+1:)=-1
      solver_cap(:np)=capacity(positive);solver_cap(np+1:)=solver_cap(:np)
      if(mode==1)then
        linear_term(:np)=epsilon-y(positive);linear_term(np+1:)=epsilon+y(positive)
      else
        linear_term(:np)=-y(positive);linear_term(np+1:)=y(positive)
        target=0
        do i=1,np
          target=target+capacity(positive(i))*nu
        end do
        remaining=target/2
        do i=1,np
          initial(i)=min(remaining,capacity(positive(i)));initial(i+np)=initial(i)
          remaining=remaining-initial(i)
        end do
      end if
      do i=1,2*np
        a=positive(1+mod(i-1,np));qd(i)=gram(a,a)
        do j=1,2*np
          b=positive(1+mod(j-1,np))
          ! SVR_Q rounds the unsigned row first, then applies exact +/- signs.
          qcache(j,i)=ones(i)*ones(j)*real(real(gram(a,b),real32),dp)
        end do
      end do
      select type(self)
      class is(svr)
        call libsvm_solve(qcache,qd,linear_term,ones,solver_cap,initial,self%tol,self%max_iter, &
          self%shrinking,rho,self%n_iter_,self%converged_)
      class is(nu_svr)
        call libsvm_solve_nu(qcache,qd,linear_term,ones,solver_cap,initial,self%tol,self%max_iter, &
          self%shrinking,rho,nu_level,self%n_iter_,self%converged_)
      class default
        call require(.false.,'internal regression solver type mismatch')
      end select
      beta=0;beta(positive)=initial(:np)-initial(np+1:)
      ! Report the physical signed-dual residual using its canonical split.
      if(mode==1)then
        alpha(:n)=max(beta,0._dp);alpha(n+1:)=max(-beta,0._dp)
      else
        alpha(positive)=initial(:np);alpha(n+positive)=initial(np+1:)
        select type(self)
        class is(nu_svr)
          self%dual_variables_=reshape(alpha,[n,2]);self%epsilon_signed_=-nu_level
        end select
      end if
      g=matmul(gram,beta)-y
      call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,self%kkt_violation_)
      self%fit_status_=merge(0,1,self%converged_);levels=0;levels(1)=rho
      if(mode==2)levels(2)=-nu_level
    else
      call require(.false.,'internal kernel solver mode mismatch')
    end if
    self%dual_full_=beta; self%support_=pack([(i,i=1,n)],abs(beta)>0)
    self%n_support_=size(self%support_); self%dual_coef_=beta(self%support_)
    if(self%kernel_fit_=='precomputed') then
      self%support_vectors_=reshape([real(dp)::],[0,0])
    else if(self%sparse_fit_)then
      self%support_vectors_csr_=sparse_rows(self%x_fit_csr_,self%support_)
    else
      self%support_vectors_=self%x_fit_(self%support_,:)
    end if
    if(self%kernel_fit_=='linear')then
      if(self%sparse_fit_)then
        allocate(self%coef_(self%x_fit_csr_%n_cols));self%coef_=0
        do i=1,n
          do j=self%x_fit_csr_%indptr(i),self%x_fit_csr_%indptr(i+1)-1
            a=self%x_fit_csr_%indices(j)
            self%coef_(a)=self%coef_(a)+beta(i)*self%x_fit_csr_%data(j)
          end do
        end do
      else
        self%coef_=matmul(beta,self%x_fit_)
      end if
    end if
  end subroutine
  subroutine svr_fit(self,x,y,sample_weight)
    class(svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%epsilon),'invalid SVR epsilon')
    call require(self%epsilon>=0,'invalid SVR epsilon'); cap=bounds(size(y),self%c,sample_weight)
    call setup(self,x,gram); call optimize(self,gram,y,cap,1,self%epsilon,0._dp,levels)
    self%intercept_=-levels(1)
  end subroutine
  subroutine nusvr_fit(self,x,y,sample_weight)
    class(nu_svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%nu),'invalid NuSVR nu')
    call require(self%nu>0.and.self%nu<=1,'invalid NuSVR nu'); cap=bounds(size(y),self%c,sample_weight)
    call setup(self,x,gram); call optimize(self,gram,y,cap,2,0._dp,self%nu,levels)
    self%intercept_=-levels(1);self%epsilon_=max(0._dp,levels(2))
  end subroutine
  subroutine one_fit(self,x,sample_weight)
    class(one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call require(ieee_is_finite(self%nu),'invalid OneClassSVM nu')
    call require(self%nu>0.and.self%nu<1,'OneClassSVM requires 0 < nu < 1')
    call setup(self,x,gram); cap=bounds(size(x,1),1._dp,sample_weight)
    call optimize(self,gram,spread(0._dp,1,size(x,1)),cap,3,0._dp,self%nu,levels)
    self%offset_=levels(1); self%intercept_=-self%offset_
  end subroutine
  subroutine svr_fit_csr(self,x,y,sample_weight)
    class(svr),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::gram(:,:),cap(:)
    real(dp)::levels(2)
    call check_csr(x);call require(x%n_rows==size(y),'SVR CSR target count mismatch')
    call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%epsilon),'invalid SVR epsilon')
    call require(self%epsilon>=0,'invalid SVR epsilon');cap=bounds(size(y),self%c,sample_weight)
    call setup_csr(self,x,gram);call optimize(self,gram,y,cap,1,self%epsilon,0._dp,levels)
    call remap_sparse_support(self,sample_weight)
    self%intercept_=-levels(1)
  end subroutine
  subroutine nusvr_fit_csr(self,x,y,sample_weight)
    class(nu_svr),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::gram(:,:),cap(:)
    real(dp)::levels(2)
    call check_csr(x);call require(x%n_rows==size(y),'NuSVR CSR target count mismatch')
    call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%nu),'invalid NuSVR nu')
    call require(self%nu>0.and.self%nu<=1,'invalid NuSVR nu');cap=bounds(size(y),self%c,sample_weight)
    call setup_csr(self,x,gram);call optimize(self,gram,y,cap,2,0._dp,self%nu,levels)
    call remap_sparse_support(self,sample_weight)
    self%intercept_=-levels(1);self%epsilon_=max(0._dp,levels(2))
  end subroutine
  subroutine one_fit_csr(self,x,sample_weight)
    class(one_class_svm),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::gram(:,:),cap(:)
    real(dp)::levels(2)
    call require(ieee_is_finite(self%nu),'invalid OneClassSVM nu')
    call require(self%nu>0.and.self%nu<1,'OneClassSVM requires 0 < nu < 1')
    call setup_csr(self,x,gram);cap=bounds(x%n_rows,1._dp,sample_weight)
    call optimize(self,gram,spread(0._dp,1,x%n_rows),cap,3,0._dp,self%nu,levels)
    call remap_sparse_support(self,sample_weight)
    self%offset_=levels(1);self%intercept_=-self%offset_
  end subroutine
  subroutine remap_sparse_support(self,sample_weight)
    class(kernel_model),intent(inout)::self
    real(dp),optional,intent(in)::sample_weight(:)
    integer::i
    ! LIBSVM's sparse wrapper reports indices in its positive-weight problem.
    ! Support vector rows themselves still refer to the original design.
    if(present(sample_weight))then
      do i=1,size(self%support_)
        self%support_(i)=count(sample_weight(:self%support_(i))>0)
      end do
    end if
  end subroutine
  function kernel_raw_csr(self,x) result(values)
    class(kernel_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    type(csr_matrix)::query
    call require(allocated(self%dual_full_),'kernel SVM is not fitted')
    call require(self%sparse_fit_,'cannot use CSR input in kernel SVM trained on dense data')
    call check_csr(x)
    call require(x%n_cols==self%x_fit_csr_%n_cols,'kernel SVM CSR feature count mismatch')
    query=svm_sorted(x)
    values=matmul(sparse_gram(self,query,self%support_vectors_csr_,.false.),self%dual_coef_)
  end function
  function kernel_raw(self,x) result(values)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    call require(allocated(self%dual_full_),'kernel SVM is not fitted')
    if(self%sparse_fit_)then
      call check_x(x);values=kernel_raw_csr(self,csr_from_dense(x));return
    end if
    call check_xy(transpose(x),size(self%x_fit_,2))
    values=matmul(gram_matrix(self,x,self%x_fit_),self%dual_full_)
  end function
  function kernel_decision(self,x) result(values)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=kernel_raw(self,x)+self%intercept_
  end function
  function regression_predict(self,x) result(values)
    class(regression_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=self%decision_function(x)
  end function
  real(dp) function regression_score(self,x,y,sample_weight) result(value)
    class(regression_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  function one_predict(self,x) result(labels)
    class(one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    labels=merge(1,-1,self%decision_function(x)>0)
  end function
  function one_score_samples(self,x) result(values)
    class(one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=kernel_raw(self,x)
  end function
  function one_fit_predict(self,x,sample_weight) result(labels)
    class(one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: labels(:)
    call self%fit(x,sample_weight); labels=self%predict(x)
  end function
  function kernel_decision_csr(self,x) result(values)
    class(kernel_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    values=kernel_raw_csr(self,x)+self%intercept_
  end function
  function regression_predict_csr(self,x) result(values)
    class(regression_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    values=self%decision_function(x)
  end function
  real(dp) function regression_score_csr(self,x,y,sample_weight) result(value)
    class(regression_model),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  function one_predict_csr(self,x) result(labels)
    class(one_class_svm),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer,allocatable::labels(:)
    labels=merge(1,-1,self%decision_function(x)>0)
  end function
  function one_score_samples_csr(self,x) result(values)
    class(one_class_svm),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::values(:)
    values=kernel_raw_csr(self,x)
  end function
  function one_fit_predict_csr(self,x,sample_weight) result(labels)
    class(one_class_svm),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    integer,allocatable::labels(:)
    call self%fit(x,sample_weight);labels=self%predict(x)
  end function
end module
