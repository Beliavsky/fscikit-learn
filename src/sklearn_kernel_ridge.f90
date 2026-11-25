! KernelRidge: sklearn 1.6.1 kernel_ridge.py and linear_model/_ridge.py
! _solve_cholesky_kernel (BSD). Typed storage/adapters are original MIT.
module sklearn_kernel_ridge
  use sklearn_common,only:dp,require,check_x,solve_lower,solve_upper,dense_svd
  use sklearn_metrics,only:r2_score
  use sklearn_kernel_dispatch,only:pairwise_kernels
  use sklearn_sparse,only:csr_matrix
  use sklearn_gp_kernels,only:gp_kernel
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::kernel_ridge
  abstract interface
    function ridge_row_kernel(x,y)result(value)
      import dp
      real(dp),intent(in)::x(:),y(:)
      real(dp)::value
    end function
  end interface
  type::kernel_ridge
    real(dp)::alpha=1,gamma=-1,coef0=1
    integer::degree=3
    character(16)::kernel='linear'
    real(dp),allocatable::alpha_values(:),x_fit_(:,:),dual_coef_(:),dual_coef_multi_(:,:)
    type(csr_matrix),allocatable::x_fit_csr_
    type(gp_kernel)::kernel_object
    procedure(ridge_row_kernel),pointer,nopass::kernel_function=>null()
    integer::n_features_in_=0,n_samples_fit_=0,n_targets_=0
    logical::sparse_fit_=.false.,used_lstsq_=.false.
  contains
    procedure,private::fit_dv,fit_dm,fit_sv,fit_sm,fit_dv_scalar,fit_dm_scalar,fit_sv_scalar,fit_sm_scalar
    generic::fit=>fit_dv,fit_dm,fit_sv,fit_sm,fit_dv_scalar,fit_dm_scalar,fit_sv_scalar,fit_sm_scalar
    procedure,private::predict_dense,predict_sparse,predict_multi_dense,predict_multi_sparse
    generic::predict=>predict_dense,predict_sparse
    generic::predict_multi=>predict_multi_dense,predict_multi_sparse
    procedure,private::score_dv,score_dm,score_sv,score_sm
    generic::score=>score_dv,score_dm,score_sv,score_sm
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
  subroutine check_controls(self)
    class(kernel_ridge),intent(in)::self
    call require(ieee_is_finite(self%gamma),'kernel ridge gamma must be finite or -1 for None')
    call require(self%gamma>=0.or.self%gamma==-1,'kernel ridge gamma must be nonnegative or -1 for None')
    call require(self%degree>=0,'kernel ridge degree must be nonnegative')
    if(self%kernel=='callable')call require(associated(self%kernel_function),'kernel ridge callback is unset')
    if(self%kernel=='gp')call require(self%kernel_object%is_set(),'kernel ridge GP kernel is unset')
    if(self%kernel=='chi2')call require(self%gamma>=0,'kernel ridge chi2 requires an explicit gamma, as in source')
  end subroutine
  subroutine check_sparse(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'kernel ridge CSR features must be nonempty')
    call require(all(ieee_is_finite(x%data)),'kernel ridge CSR features must be finite')
  end subroutine
  function gram_dd(self,x,y)result(k)
    class(kernel_ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),allocatable::k(:,:)
    real(dp),allocatable::g
    if(self%gamma>=0)g=self%gamma
    if(self%kernel=='callable')then
      k=pairwise_kernels(x,y,self%kernel_function,filter_params=.true.)
    else if(self%kernel=='gp')then
      k=pairwise_kernels(x,y,self%kernel_object)
    else
      k=pairwise_kernels(x,y,self%kernel,g,self%degree,self%coef0,filter_params=.true.)
    end if
  end function
  function gram_ds(self,x,y)result(k)
    class(kernel_ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    type(csr_matrix),intent(in)::y
    real(dp),allocatable::k(:,:)
    real(dp),allocatable::g
    if(self%gamma>=0)g=self%gamma
    call require(self%kernel/='gp'.and.self%kernel/='precomputed','GP and precomputed kernels require dense arrays')
    if(self%kernel=='callable')then
      k=pairwise_kernels(x,y,self%kernel_function,filter_params=.true.)
    else
      k=pairwise_kernels(x,y,self%kernel,g,self%degree,self%coef0,filter_params=.true.)
    end if
  end function
  function gram_sd(self,x,y)result(k)
    class(kernel_ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),allocatable::k(:,:)
    real(dp),allocatable::g
    if(self%gamma>=0)g=self%gamma
    call require(self%kernel/='gp'.and.self%kernel/='precomputed','GP and precomputed kernels require dense arrays')
    if(self%kernel=='callable')then
      k=pairwise_kernels(x,y,self%kernel_function,filter_params=.true.)
    else
      k=pairwise_kernels(x,y,self%kernel,g,self%degree,self%coef0,filter_params=.true.)
    end if
  end function
  function gram_ss(self,x,y)result(k)
    class(kernel_ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    type(csr_matrix),intent(in)::y
    real(dp),allocatable::k(:,:)
    real(dp),allocatable::g
    if(self%gamma>=0)g=self%gamma
    call require(self%kernel/='gp'.and.self%kernel/='precomputed','GP and precomputed kernels require dense arrays')
    if(self%kernel=='callable')then
      k=pairwise_kernels(x,y,self%kernel_function,filter_params=.true.)
    else
      k=pairwise_kernels(x,y,self%kernel,g,self%degree,self%coef0,filter_params=.true.)
    end if
  end function
  function train_gram_dense(self,x)result(k)
    class(kernel_ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::k(:,:),g
    if(self%gamma>=0)g=self%gamma
    if(self%kernel=='callable')then
      k=pairwise_kernels(x,metric=self%kernel_function,filter_params=.true.)
    else if(self%kernel=='gp')then
      k=pairwise_kernels(x,metric=self%kernel_object)
    else
      k=pairwise_kernels(x,metric=self%kernel,gamma=g,degree=self%degree,coef0=self%coef0,filter_params=.true.)
    end if
  end function
  function train_gram_sparse(self,x)result(k)
    class(kernel_ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::k(:,:),g
    if(self%gamma>=0)g=self%gamma
    call require(self%kernel/='gp'.and.self%kernel/='precomputed','GP and precomputed kernels require dense arrays')
    if(self%kernel=='callable')then
      k=pairwise_kernels(x,metric=self%kernel_function,filter_params=.true.)
    else
      k=pairwise_kernels(x,metric=self%kernel,gamma=g,degree=self%degree,coef0=self%coef0,filter_params=.true.)
    end if
  end function
  subroutine fit_dv(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::k(:,:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call check_controls(self)
    call check_x(x)
    call require(size(y,1)==size(x,1),'kernel ridge sample counts differ')
    call check_x(reshape(y,[size(y),1]))
    call fp_begin(modes,flags)
    k=train_gram_dense(self,x)
    call fit_system(self,k,reshape(y,[size(y),1]),sample_weight)
    self%n_features_in_=size(x,2);self%n_samples_fit_=size(x,1);self%sparse_fit_=.false.
    self%x_fit_=x
    if(allocated(self%x_fit_csr_))deallocate(self%x_fit_csr_)
    call fp_end(modes,flags)
  end subroutine
  subroutine fit_dv_scalar(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:),sample_weight
    call require(ieee_is_finite(sample_weight).and.sample_weight>=0,'invalid scalar kernel ridge weight')
    call self%fit_dv(x,y,spread(sample_weight,1,size(x,1)))
  end subroutine
  subroutine fit_dm(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::k(:,:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call check_controls(self)
    call check_x(x)
    call require(size(y,1)==size(x,1),'kernel ridge sample counts differ')
    call check_x(y)
    call fp_begin(modes,flags)
    k=train_gram_dense(self,x)
    call fit_system(self,k,y,sample_weight)
    self%n_features_in_=size(x,2);self%n_samples_fit_=size(x,1);self%sparse_fit_=.false.
    self%x_fit_=x
    if(allocated(self%x_fit_csr_))deallocate(self%x_fit_csr_)
    call fp_end(modes,flags)
  end subroutine
  subroutine fit_dm_scalar(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),sample_weight
    call require(ieee_is_finite(sample_weight).and.sample_weight>=0,'invalid scalar kernel ridge weight')
    call self%fit_dm(x,y,spread(sample_weight,1,size(x,1)))
  end subroutine
  function predict_dense(self,x)result(y)
    class(kernel_ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:),matrix(:,:)
    call require(allocated(self%dual_coef_multi_),'kernel ridge is not fitted')
    call require(self%n_targets_==1,'use predict_multi for multi-target kernel ridge')
    matrix=self%predict_multi_dense(x);y=matrix(:,1)
  end function
  function predict_multi_dense(self,x)result(y)
    class(kernel_ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:,:),k(:,:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call require(allocated(self%dual_coef_multi_),'kernel ridge is not fitted')
    call check_x(x)
    if(self%kernel=='precomputed')then
      call require(size(x,2)==self%n_samples_fit_,'kernel ridge precomputed query width mismatch')
    else
      call require(size(x,2)==self%n_features_in_,'kernel ridge query feature count mismatch')
    end if
    call check_controls(self);call fp_begin(modes,flags)
    if(self%sparse_fit_)then
      k=gram_ds(self,x,self%x_fit_csr_)
    else
      k=gram_dd(self,x,self%x_fit_)
    end if
    y=matmul(k,self%dual_coef_multi_)
    call fp_end(modes,flags)
  end function
  real(dp)function score_dv(self,x,y,sample_weight)result(value)
    class(kernel_ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function score_dm(self,x,y,sample_weight)result(value)
    class(kernel_ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict_multi(x),sample_weight=sample_weight)
  end function
  subroutine fit_sv(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::k(:,:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call check_controls(self)
    call check_sparse(x)
    call require(size(y,1)==x%n_rows,'kernel ridge sample counts differ')
    call check_x(reshape(y,[size(y),1]))
    call fp_begin(modes,flags)
    k=train_gram_sparse(self,x)
    call fit_system(self,k,reshape(y,[size(y),1]),sample_weight)
    self%n_features_in_=x%n_cols;self%n_samples_fit_=x%n_rows;self%sparse_fit_=.true.
    self%x_fit_csr_=x
    if(allocated(self%x_fit_))deallocate(self%x_fit_)
    call fp_end(modes,flags)
  end subroutine
  subroutine fit_sv_scalar(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),sample_weight
    call require(ieee_is_finite(sample_weight).and.sample_weight>=0,'invalid scalar kernel ridge weight')
    call self%fit_sv(x,y,spread(sample_weight,1,x%n_rows))
  end subroutine
  subroutine fit_sm(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::k(:,:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call check_controls(self)
    call check_sparse(x)
    call require(size(y,1)==x%n_rows,'kernel ridge sample counts differ')
    call check_x(y)
    call fp_begin(modes,flags)
    k=train_gram_sparse(self,x)
    call fit_system(self,k,y,sample_weight)
    self%n_features_in_=x%n_cols;self%n_samples_fit_=x%n_rows;self%sparse_fit_=.true.
    self%x_fit_csr_=x
    if(allocated(self%x_fit_))deallocate(self%x_fit_)
    call fp_end(modes,flags)
  end subroutine
  subroutine fit_sm_scalar(self,x,y,sample_weight)
    class(kernel_ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),sample_weight
    call require(ieee_is_finite(sample_weight).and.sample_weight>=0,'invalid scalar kernel ridge weight')
    call self%fit_sm(x,y,spread(sample_weight,1,x%n_rows))
  end subroutine
  function predict_sparse(self,x)result(y)
    class(kernel_ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:),matrix(:,:)
    call require(allocated(self%dual_coef_multi_),'kernel ridge is not fitted')
    call require(self%n_targets_==1,'use predict_multi for multi-target kernel ridge')
    matrix=self%predict_multi_sparse(x);y=matrix(:,1)
  end function
  function predict_multi_sparse(self,x)result(y)
    class(kernel_ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:,:),k(:,:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call require(allocated(self%dual_coef_multi_),'kernel ridge is not fitted')
    call check_sparse(x)
    if(self%kernel=='precomputed')then
      call require(x%n_cols==self%n_samples_fit_,'kernel ridge precomputed query width mismatch')
    else
      call require(x%n_cols==self%n_features_in_,'kernel ridge query feature count mismatch')
    end if
    call check_controls(self);call fp_begin(modes,flags)
    if(self%sparse_fit_)then
      k=gram_ss(self,x,self%x_fit_csr_)
    else
      k=gram_sd(self,x,self%x_fit_)
    end if
    y=matmul(k,self%dual_coef_multi_)
    call fp_end(modes,flags)
  end function
  real(dp)function score_sv(self,x,y,sample_weight)result(value)
    class(kernel_ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function score_sm(self,x,y,sample_weight)result(value)
    class(kernel_ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict_multi(x),sample_weight=sample_weight)
  end function
  subroutine fit_system(self,k,y,weights)
    class(kernel_ridge),intent(inout)::self
    real(dp),intent(inout)::k(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::weights(:)
    real(dp),allocatable::penalty(:),rhs(:,:),sw(:),coef(:,:),target(:,:)
    integer::n,t,i,j
    logical::one_alpha,okay
    n=size(y,1);t=size(y,2)
    call require(all(shape(k)==[n,n]),'kernel ridge training kernel must be square')
    call require(all(ieee_is_finite(k)),'kernel ridge training kernel exceeds finite working range')
    allocate(penalty(t));penalty=self%alpha
    if(allocated(self%alpha_values))then
      call require(size(self%alpha_values)>0,'kernel ridge alpha_values must be nonempty')
      call require(size(self%alpha_values)==1.or.size(self%alpha_values)==t,'kernel ridge penalty count mismatch')
      penalty=self%alpha_values(1)
      if(size(self%alpha_values)==t)penalty=self%alpha_values
    end if
    call require(all(ieee_is_finite(penalty)).and.all(penalty>=0),'kernel ridge penalties must be finite and nonnegative')
    rhs=y;allocate(sw(n));sw=1
    if(present(weights))then
      call require(size(weights)==n,'kernel ridge weight count mismatch')
      call require(all(ieee_is_finite(weights)).and.all(weights>=0),'kernel ridge weights must be finite and nonnegative')
      sw=sqrt(weights);rhs=rhs*spread(sw,2,t)
      do j=1,n
        k(:,j)=k(:,j)*(sw*sw(j))
      end do
    end if
    call require(all(ieee_is_finite(k)).and.all(ieee_is_finite(rhs)),'weighted kernel ridge exceeds finite working range')
    one_alpha=all(penalty==penalty(1));self%used_lstsq_=.false.
    if(one_alpha)then
      do i=1,n
        k(i,i)=k(i,i)+penalty(1)
      end do
      call positive_solve(k,rhs,coef,okay)
      if(.not.okay)then
        call least_squares(k,rhs,coef);self%used_lstsq_=.true.
      end if
    else
      allocate(coef(n,t))
      do j=1,t
        do i=1,n
          k(i,i)=k(i,i)+penalty(j)
        end do
        call positive_solve(k,rhs(:,j:j),target,okay)
        call require(okay,'kernel ridge nonuniform penalties require positive definite target systems')
        coef(:,j)=target(:,1)
        do i=1,n
          k(i,i)=k(i,i)-penalty(j)
        end do
      end do
    end if
    if(present(weights))coef=coef*spread(sw,2,t)
    call require(all(ieee_is_finite(coef)),'kernel ridge solution exceeds finite working range')
    self%dual_coef_multi_=coef;self%dual_coef_=coef(:,1);self%n_targets_=t
  end subroutine
  subroutine positive_solve(a,b,x,okay)
    ! Source assume_a='pos' consumes the UPPER triangle. The native Cholesky
    ! convention is shared with ridge_solver; multiple RHS reuse one factor.
    real(dp),intent(in)::a(:,:),b(:,:)
    real(dp),allocatable,intent(out)::x(:,:)
    logical,intent(out)::okay
    real(dp)::l(size(a,1),size(a,1)),diagonal
    integer::i,j
    l=0;okay=.false.
    do i=1,size(a,1)
      do j=1,i-1
        l(i,j)=(a(j,i)-dot_product(l(i,:j-1),l(j,:j-1)))/l(j,j)
      end do
      diagonal=a(i,i)-sum(l(i,:i-1)**2)
      if(.not.ieee_is_finite(diagonal).or.diagonal<=0)return
      l(i,i)=sqrt(diagonal)
    end do
    allocate(x(size(b,1),size(b,2)))
    do j=1,size(b,2)
      x(:,j)=solve_upper(transpose(l),solve_lower(l,b(:,j)))
    end do
    okay=all(ieee_is_finite(x))
  end subroutine
  subroutine least_squares(a,b,x)
    real(dp),intent(in)::a(:,:),b(:,:)
    real(dp),allocatable,intent(out)::x(:,:)
    real(dp),allocatable::u(:,:),v(:,:),s(:)
    integer::n,j,k
    n=size(a,1);allocate(u(n,n),v(n,n),s(n),x(n,size(b,2)))
    call dense_svd(a,s,u,v);x=0
    do k=1,n
      if(s(k)<=epsilon(1._dp)*maxval(s))cycle
      do j=1,size(b,2)
        x(:,j)=x(:,j)+v(:,k)*(dot_product(u(:,k),b(:,j))/s(k))
      end do
    end do
  end subroutine
end module
