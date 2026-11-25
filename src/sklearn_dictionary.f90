! Native adaptation of sklearn 1.6.1 decomposition/_dict_learning.py and _sparse_pca.py.
! BSD-3-Clause upstream notices are retained in LICENSES and THIRD_PARTY_NOTICES.md.
module sklearn_dictionary
  use sklearn_kernel_dispatch,only:kernel_blas_dot
  use sklearn_randomized_svd_core, only: randomized_svd
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_common
  use sklearn_least_angle, only: lars_path,lars_path_result
  use sklearn_linear_model, only: orthogonal_matching_pursuit
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: sparse_pca,dictionary_learning,sparse_coder,sparse_encode
  public :: mini_batch_dictionary_learning,mini_batch_sparse_pca
  public :: dict_learning,dict_learning_online,dictionary_learning_result,online_dictionary_result
  public :: dictionary_callback,online_dictionary_callback
  abstract interface
    subroutine dictionary_callback(iteration,code,dictionary,errors)
      import dp
      integer, intent(in) :: iteration
      real(dp), intent(inout) :: code(:,:),dictionary(:,:),errors(:)
    end subroutine
    subroutine online_dictionary_callback(step,cost,dictionary,a,b)
      import dp,int64
      integer(int64), intent(in) :: step
      real(dp), intent(in) :: cost
      real(dp), intent(inout) :: dictionary(:,:),a(:,:),b(:,:)
    end subroutine
  end interface
  type :: dictionary_learning_result
    real(dp), allocatable :: code(:,:),dictionary(:,:),errors(:)
    integer :: n_iter=0
  end type
  type :: online_dictionary_result
    real(dp), allocatable :: code(:,:),dictionary(:,:)
    integer(int64) :: n_iter=0,n_steps=0
  end type
  type :: coding_parameters
    character(12) :: transform_algorithm='omp'
    integer :: transform_n_nonzero_coefs=0,transform_max_iter=1000
    real(dp) :: transform_alpha=-1
    logical :: split_sign=.false.,positive_code=.false.
  end type
  type, extends(coding_parameters) :: dictionary_learning
    integer :: n_components=0,max_iter=1000,random_state=0,n_iter_=0,n_features_in_=0
    real(dp) :: alpha=1,tol=1.e-8_dp
    character(4) :: fit_algorithm='lars'
    logical :: positive_dict=.false.
    procedure(dictionary_callback), pointer, nopass :: callback=>null()
    real(dp), allocatable :: components_(:,:),error_(:),dict_init(:,:),code_init(:,:)
  contains
    procedure :: fit=>dictionary_fit
    procedure :: fit_transform=>dictionary_fit_transform
    procedure :: transform=>dictionary_transform
  end type
  type, extends(coding_parameters) :: sparse_coder
    real(dp), allocatable :: dictionary(:,:)
  contains
    procedure :: fit=>coder_fit
    procedure :: fit_transform=>coder_fit_transform
    procedure :: transform=>coder_transform
  end type
  type :: sparse_pca
    integer :: n_components=0,n_components_=0,max_iter=1000,random_state=0,n_iter_=0,n_features_in_=0
    real(dp) :: alpha=1,ridge_alpha=.01_dp,tol=1.e-8_dp
    character(4) :: method='lars'
    real(dp), allocatable :: mean_(:),components_(:,:),error_(:),u_init(:,:),v_init(:,:)
  contains
    procedure :: fit=>sparse_pca_fit
    procedure :: fit_transform=>sparse_pca_fit_transform
    procedure :: transform=>sparse_pca_transform
    procedure :: inverse_transform=>sparse_pca_inverse
  end type
  type, extends(coding_parameters) :: mini_batch_dictionary_learning
    integer :: n_components=0,n_components_=0,batch_size=256,batch_size_=0,random_state=0,n_features_in_=0
    integer :: max_iter=1000,max_no_improvement=10
    integer(int64) :: n_iter_=0,n_steps_=0,no_improvement_=0
    real(dp) :: alpha=1,tol=.001_dp,ewa_cost_=0,ewa_cost_min_=0
    character(4) :: fit_algorithm='lars'
    logical :: positive_dict=.false.,shuffle=.true.
    procedure(online_dictionary_callback), pointer, nopass :: callback=>null()
    real(dp), allocatable :: components_(:,:),a_(:,:),b_(:,:),dict_init(:,:)
    type(legacy_random_state), private :: rng
    character(4), private :: fitted_algorithm='lars'
  contains
    procedure :: fit=>mini_dictionary_fit
    procedure :: partial_fit=>mini_dictionary_partial_fit
    procedure :: fit_transform=>mini_dictionary_fit_transform
    procedure :: transform=>mini_dictionary_transform
  end type
  type :: mini_batch_sparse_pca
    integer :: n_components=0,n_components_=0,batch_size=3,max_iter=1000,max_no_improvement=10,random_state=0,n_features_in_=0
    integer(int64) :: n_iter_=0,n_steps_=0
    real(dp) :: alpha=1,ridge_alpha=.01_dp,tol=.001_dp
    character(4) :: method='lars'
    logical :: shuffle=.true.
    real(dp), allocatable :: mean_(:),components_(:,:)
  contains
    procedure :: fit=>mini_sparse_pca_fit
    procedure :: fit_transform=>mini_sparse_pca_fit_transform
    procedure :: transform=>mini_sparse_pca_transform
    procedure :: inverse_transform=>mini_sparse_pca_inverse
  end type
  type :: learning_result
    real(dp), allocatable :: code(:,:),dictionary(:,:),errors(:)
    integer :: n_iter=0
  end type
contains
  function dict_learning(x,n_components,alpha,max_iter,tol,method,dict_init,code_init, &
      callback,random_state,positive_dict,positive_code,method_max_iter) result(result)
    real(dp), intent(in) :: x(:,:),alpha
    integer, intent(in) :: n_components
    integer, optional, intent(in) :: max_iter,random_state,method_max_iter
    real(dp), optional, intent(in) :: tol,dict_init(:,:),code_init(:,:)
    character(*), optional, intent(in) :: method
    logical, optional, intent(in) :: positive_dict,positive_code
    procedure(dictionary_callback), optional :: callback
    type(dictionary_learning_result) :: result
    type(dictionary_learning) :: model
    model%n_components=n_components;model%alpha=alpha;model%max_iter=100
    if(present(max_iter))model%max_iter=max_iter
    if(present(tol))model%tol=tol
    if(present(method))then
      call require(method=='cd'.or.method=='lars','unknown dictionary fit algorithm')
      model%fit_algorithm=method
    end if
    if(present(dict_init))model%dict_init=dict_init
    if(present(code_init))model%code_init=code_init
    if(present(callback))model%callback=>callback
    if(present(random_state))model%random_state=random_state
    if(present(positive_dict))model%positive_dict=positive_dict
    if(present(positive_code))model%positive_code=positive_code
    if(present(method_max_iter))model%transform_max_iter=method_max_iter
    result%code=model%fit_transform(x);result%dictionary=model%components_
    result%errors=model%error_;result%n_iter=model%n_iter_
  end function
  function dict_learning_online(x,n_components,alpha,max_iter,return_code,dict_init,callback, &
      batch_size,shuffle,method,random_state,positive_dict,positive_code,method_max_iter,tol, &
      max_no_improvement) result(result)
    real(dp), intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_components,max_iter,batch_size,random_state,method_max_iter,max_no_improvement
    real(dp), optional, intent(in) :: alpha,dict_init(:,:),tol
    logical, optional, intent(in) :: return_code,shuffle,positive_dict,positive_code
    character(*), optional, intent(in) :: method
    procedure(online_dictionary_callback), optional :: callback
    type(online_dictionary_result) :: result
    type(mini_batch_dictionary_learning) :: model
    logical :: codes
    model%n_components=2;model%max_iter=100;codes=.true.
    if(present(n_components))model%n_components=n_components
    if(present(alpha))model%alpha=alpha
    if(present(max_iter))model%max_iter=max_iter
    if(present(return_code))codes=return_code
    if(present(dict_init))model%dict_init=dict_init
    if(present(callback))model%callback=>callback
    if(present(batch_size))model%batch_size=batch_size
    if(present(shuffle))model%shuffle=shuffle
    if(present(method))then
      call require(method=='cd'.or.method=='lars','unknown dictionary fit algorithm')
      model%fit_algorithm=method
    end if
    if(present(random_state))model%random_state=random_state
    if(present(positive_dict))model%positive_dict=positive_dict
    if(present(positive_code))model%positive_code=positive_code
    if(present(method_max_iter))model%transform_max_iter=method_max_iter
    if(present(tol))model%tol=tol
    if(present(max_no_improvement))model%max_no_improvement=max_no_improvement
    model%transform_algorithm='lasso_'//trim(model%fit_algorithm);model%transform_alpha=model%alpha
    call model%fit(x)
    result%dictionary=model%components_;result%n_iter=model%n_iter_;result%n_steps=model%n_steps_
    if(codes)result%code=model%transform(x)
  end function
  recursive real(dp) function dictionary_numpy_sum(values) result(total)
    ! NumPy 2.2.6 loops_utils.h.src pairwise reduction; retained NumPy BSD notice.
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,split,whole
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128) then
      partial=values(:8);whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=dictionary_numpy_sum(values(:split))+dictionary_numpy_sum(values(split+1:))
    end if
  end function
  real(dp) function dictionary_objective(x,code,dictionary,alpha) result(cost)
    real(dp),intent(in)::x(:,:),code(:,:),dictionary(:,:),alpha
    real(dp),allocatable::residual(:,:)
    integer::i,j
    allocate(residual(size(x,1),size(x,2)))
    do j=1,size(x,2)
      do i=1,size(x,1)
        residual(i,j)=x(i,j)-kernel_blas_dot(code(i,:),dictionary(:,j),size(x,1),size(x,2),i,j)
      end do
    end do
    cost=.5_dp*dictionary_numpy_sum(reshape(transpose(residual**2),[size(residual)])) &
      +alpha*dictionary_numpy_sum(reshape(transpose(abs(code)),[size(code)]))
  end function
  logical function valid_coding_algorithm(algorithm) result(valid)
    character(*), intent(in) :: algorithm
    valid=algorithm=='lasso_lars'.or.algorithm=='lasso_cd'.or.algorithm=='lars'.or. &
      algorithm=='omp'.or.algorithm=='threshold'
  end function
  subroutine coding_cd(gram,row,ynorm,alpha,positive,limit,w)
    real(dp), intent(in) :: gram(:,:),row(:),ynorm,alpha
    logical, intent(in) :: positive
    integer, intent(in) :: limit
    real(dp), intent(inout) :: w(:)
    real(dp) :: h(size(w)),old,value,change,largest,dual,constant,rnorm,gap,product
    real(dp), parameter :: tolerance=1.e-4_dp
    integer :: it,j
    h=matmul(gram,w)
    do it=1,limit
      change=0; largest=0
      do j=1,size(w)
        if(gram(j,j)==0) cycle
        old=w(j); h=h-old*gram(:,j); value=row(j)-h(j)
        w(j)=sign(max(abs(value)-alpha,0._dp),value)/gram(j,j)
        if(positive.and.value<0) w(j)=0
        h=h+w(j)*gram(:,j)
        change=max(change,abs(w(j)-old)); largest=max(largest,abs(w(j)))
      end do
      if(largest>0) then
        if(change/largest>=tolerance.and.it<limit) cycle
      end if
      product=dot_product(w,row); rnorm=ynorm+dot_product(w,h)-2*product
      dual=maxval(abs(row-h)); if(positive) dual=maxval(row-h)
      constant=1; if(dual>alpha) constant=alpha/dual
      gap=.5_dp*rnorm*(1+constant**2)+alpha*sum(abs(w))-constant*ynorm+constant*product
      if(gap<tolerance*ynorm) exit
    end do
  end subroutine
  function sparse_encode(x,dictionary,algorithm,alpha,n_nonzero_coefs,max_iter,positive,init) result(code)
    real(dp), intent(in) :: x(:,:),dictionary(:,:)
    character(*), intent(in), optional :: algorithm
    real(dp), intent(in), optional :: alpha,init(:,:)
    integer, intent(in), optional :: n_nonzero_coefs,max_iter
    logical, intent(in), optional :: positive
    real(dp), allocatable :: code(:,:),gram(:,:),cov(:,:)
    type(lars_path_result) :: path
    type(orthogonal_matching_pursuit) :: omp
    real(dp) :: penalty
    integer :: i,k,width,limit
    logical :: constrained
    character(12) :: method
    call check_x(x); call check_x(dictionary)
    call require(size(x,2)==size(dictionary,2),'sparse coding feature count mismatch')
    method='lasso_lars'
    if(present(algorithm)) then
      call require(valid_coding_algorithm(algorithm),'unknown sparse coding algorithm'); method=algorithm
    end if
    penalty=1; if(present(alpha)) penalty=alpha
    call require(ieee_is_finite(penalty),'sparse coding alpha must be finite and nonnegative')
    call require(penalty>=0,'sparse coding alpha must be finite and nonnegative')
    limit=1000; if(present(max_iter)) limit=max_iter
    call require(limit>=0.and.limit<huge(limit),'invalid sparse coding max_iter')
    k=size(dictionary,1); width=min(max(1,size(x,2)/10),k)
    if(present(n_nonzero_coefs)) then
      call require(n_nonzero_coefs>=0,'sparse coding sparsity must be nonnegative (zero means auto)')
      if(n_nonzero_coefs>0) width=n_nonzero_coefs
    end if
    if(method=='omp') call require(width<=k,'OMP sparse coding count exceeds dictionary size')
    constrained=.false.; if(present(positive)) constrained=positive
    call require(.not.(constrained.and.(method=='omp'.or.method=='lars')), &
      'positive sparse coding requires lasso or threshold')
    allocate(code(size(x,1),k)); code=0
    if(present(init)) then
      call require(all(shape(init)==shape(code)),'sparse coding initializer shape mismatch')
      call require(all(ieee_is_finite(init)),'nonfinite sparse coding initializer')
      if(method=='lasso_cd') code=init
    end if
    gram=matmul(dictionary,transpose(dictionary)); cov=matmul(x,transpose(dictionary))
    do i=1,size(x,1)
      select case(trim(method))
      case('lasso_lars')
        path=lars_path(transpose(dictionary),x(i,:),method='lasso',alpha_min=penalty/size(x,2), &
          max_iter=limit,positive=constrained)
        code(i,:)=path%coefficients(:,size(path%alphas))
      case('lasso_cd')
        call coding_cd(gram,cov(i,:),sum(x(i,:)**2),penalty,constrained,limit,code(i,:))
      case('lars')
        path=lars_path(transpose(dictionary),x(i,:),method='lar',max_iter=width)
        code(i,:)=path%coefficients(:,size(path%alphas))
      case('omp')
        omp%fit_intercept=.false.; omp%n_nonzero_coefs=width
        call omp%fit(transpose(dictionary),x(i,:)); code(i,:)=omp%coef_
      case('threshold')
        code(i,:)=sign(max(abs(cov(i,:))-penalty,0._dp),cov(i,:))
        if(constrained) code(i,:)=max(code(i,:),0._dp)
      end select
    end do
    call require(all(ieee_is_finite(code)),'nonfinite sparse codes')
  end function
  subroutine update_dictionary(dictionary,y,code,a,b,rng,positive_dict)
    real(dp), intent(inout) :: dictionary(:,:),code(:,:)
    real(dp), intent(in) :: y(:,:),a(:,:),b(:,:)
    type(legacy_random_state), intent(inout) :: rng
    logical, intent(in) :: positive_dict
    real(dp), allocatable :: sample(:)
    real(dp) :: scale,norm
    integer :: n,p,k,j,index
    n=size(y,1); p=size(y,2); k=size(dictionary,1)
    do j=1,k
      if(a(j,j)>1.e-6_dp) then
        dictionary(j,:)=dictionary(j,:)+(b(:,j)-matmul(a(j,:),dictionary))/a(j,j)
      else
        index=rng%integer_index(n); sample=y(index,:)
        scale=sqrt(sum((sample-sum(sample)/p)**2)/p); if(scale==0) scale=1
        scale=.01_dp*scale
        do index=1,p
          sample(index)=sample(index)+scale*rng%normal()
        end do
        dictionary(j,:)=sample; code(:,j)=0
      end if
      if(positive_dict) dictionary(j,:)=max(dictionary(j,:),0._dp)
      norm=sqrt(sum(dictionary(j,:)**2)); dictionary(j,:)=dictionary(j,:)/max(norm,1._dp)
    end do
  end subroutine
  function learn_dictionary(y,k,alpha,limit,tolerance,method,inner_limit,seed,positive_dict, &
      positive_code,code_init,dict_init,callback) result(model)
    real(dp), intent(in) :: y(:,:),alpha,tolerance
    integer, intent(in) :: k,limit,inner_limit,seed
    character(*), intent(in) :: method
    logical, intent(in) :: positive_dict,positive_code
    real(dp), intent(in), optional :: code_init(:,:),dict_init(:,:)
    type(learning_result) :: model
    procedure(dictionary_callback), optional :: callback
    type(legacy_random_state) :: rng
    real(dp), allocatable :: u(:,:),v(:,:),s(:),a(:,:),b(:,:),history(:)
    real(dp) :: cost
    integer :: n,p,r,j,it,index
    call check_x(y); n=size(y,1); p=size(y,2)
    call require(k>=1,'dictionary component count must be positive')
    call require(limit>=0.and.limit<huge(limit).and.inner_limit>=0.and.inner_limit<huge(inner_limit), &
      'invalid dictionary iteration limit')
    call require(all(ieee_is_finite([alpha,tolerance])), &
      'dictionary alpha and tolerance must be finite and nonnegative')
    call require(alpha>=0.and.tolerance>=0,'dictionary alpha and tolerance must be finite and nonnegative')
    call require(method=='lars'.or.method=='cd','unknown dictionary fit algorithm')
    call require(.not.(positive_code.and.method=='lars'),'positive dictionary codes require fit_algorithm cd')
    call rng%seed(seed)
    allocate(model%code(n,k),model%dictionary(k,p),history(limit)); model%code=0; model%dictionary=0
    if(present(code_init).and.present(dict_init)) then
      call require(size(code_init,1)==n.and.size(dict_init,2)==p.and.size(code_init,2)==size(dict_init,1), &
        'dictionary initializer shape mismatch')
      call require(all(ieee_is_finite(code_init)).and.all(ieee_is_finite(dict_init)), &
        'nonfinite dictionary initializer')
      r=min(k,size(dict_init,1)); model%code(:,:r)=code_init(:,:r); model%dictionary(:r,:)=dict_init(:r,:)
    else
      allocate(u(n,p),v(p,p),s(p)); call dense_svd(y,s,u,v); r=min(k,n,p)
      model%code(:,:r)=u(:,:r); model%dictionary(:r,:)=transpose(v(:,:r))*spread(s(:r),2,p)
      if(all(y==0))then
        ! LAPACK's zero-matrix SVD returns leading identity columns. CD keeps
        ! these warm coefficients when the corresponding Gram diagonal is zero.
        model%code=0
        do j=1,r
          model%code(j,j)=1
        end do
      end if
      do j=1,r
        index=maxloc(abs(model%code(:,j)),dim=1)
        if(model%code(index,j)<0) then
          model%code(:,j)=-model%code(:,j); model%dictionary(j,:)=-model%dictionary(j,:)
        end if
      end do
    end if
    do it=1,limit
      model%code=sparse_encode(y,model%dictionary,algorithm='lasso_'//trim(method),alpha=alpha, &
        max_iter=inner_limit,positive=positive_code,init=model%code)
      a=matmul(transpose(model%code),model%code); b=matmul(transpose(y),model%code)
      call update_dictionary(model%dictionary,y,model%code,a,b,rng,positive_dict)
      cost=dictionary_objective(y,model%code,model%dictionary,alpha)
      history(it)=cost; model%n_iter=it
      if(it>1) then
        if(history(it-1)-cost<tolerance*cost) exit
      end if
      if(mod(it-1,5)==0.and.present(callback))call callback(it-1,model%code,model%dictionary,history(:it))
    end do
    model%errors=history(:model%n_iter)
  end function
  subroutine check_coding_parameters(self)
    class(coding_parameters), intent(in) :: self
    call require(valid_coding_algorithm(self%transform_algorithm),'unknown sparse coding algorithm')
    call require(self%transform_n_nonzero_coefs>=0,'sparse coding sparsity must be nonnegative (zero means auto)')
    call require(self%transform_max_iter>=0.and.self%transform_max_iter<huge(self%transform_max_iter), &
      'invalid sparse coding max_iter')
    call require(ieee_is_finite(self%transform_alpha),'transform alpha must be -1 (auto) or nonnegative')
    call require(self%transform_alpha==-1.or.self%transform_alpha>=0,'transform alpha must be -1 (auto) or nonnegative')
  end subroutine
  function apply_coding(self,x,dictionary,default_alpha) result(code)
    class(coding_parameters), intent(in) :: self
    real(dp), intent(in) :: x(:,:),dictionary(:,:),default_alpha
    real(dp), allocatable :: code(:,:),positive_part(:,:)
    real(dp) :: penalty
    call check_coding_parameters(self); penalty=self%transform_alpha
    if(penalty==-1) penalty=default_alpha
    code=sparse_encode(x,dictionary,algorithm=trim(self%transform_algorithm),alpha=penalty, &
      n_nonzero_coefs=self%transform_n_nonzero_coefs,max_iter=self%transform_max_iter,positive=self%positive_code)
    if(self%split_sign) then
      positive_part=max(code,0._dp); code=reshape([reshape(positive_part,[size(code)]), &
        reshape(-min(code,0._dp),[size(code)])],[size(code,1),2*size(code,2)])
    end if
  end function
  subroutine dictionary_fit(self,x)
    class(dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: code(:,:)
    code=self%fit_transform(x)
  end subroutine
  function dictionary_fit_transform(self,x) result(code)
    class(dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: code(:,:)
    type(learning_result) :: model
    integer :: k
    call check_x(x); call check_coding_parameters(self)
    call require(self%n_components>=0,'dictionary n_components must be nonnegative (zero means auto)')
    k=self%n_components; if(k==0) k=size(x,2)
    model=learn_dictionary(x,k,self%alpha,self%max_iter,self%tol,trim(self%fit_algorithm), &
      self%transform_max_iter,self%random_state,self%positive_dict,self%positive_code,self%code_init,self%dict_init,self%callback)
    self%components_=model%dictionary; self%error_=model%errors; self%n_iter_=model%n_iter
    self%n_features_in_=size(x,2); code=model%code
  end function
  function dictionary_transform(self,x) result(code)
    class(dictionary_learning), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: code(:,:)
    call require(allocated(self%components_),'DictionaryLearning is not fitted')
    code=apply_coding(self,x,self%components_,self%alpha)
  end function
  subroutine coder_fit(self,x)
    class(sparse_coder), intent(inout) :: self
    real(dp), intent(in), optional :: x(:,:)
    ! SparseCoder has no learned state; sklearn's fit deliberately ignores X.
  end subroutine
  function coder_fit_transform(self,x) result(code)
    class(sparse_coder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: code(:,:)
    code=self%transform(x)
  end function
  function coder_transform(self,x) result(code)
    class(sparse_coder), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: code(:,:)
    call require(allocated(self%dictionary),'SparseCoder dictionary is not allocated')
    code=apply_coding(self,x,self%dictionary,1._dp)
  end function
  subroutine sparse_pca_fit(self,x)
    class(sparse_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(learning_result) :: model
    real(dp), allocatable :: centered(:,:)
    real(dp) :: norm
    integer :: k,j,index
    call check_x(x)
    call require(self%n_components>=0,'SparsePCA n_components must be nonnegative (zero means auto)')
    call require(ieee_is_finite(self%ridge_alpha),'SparsePCA ridge_alpha must be nonnegative')
    call require(self%ridge_alpha>=0,'SparsePCA ridge_alpha must be nonnegative')
    k=self%n_components; if(k==0) k=size(x,2)
    self%mean_=sum(x,dim=1)/size(x,1); centered=x-spread(self%mean_,1,size(x,1))
    if(allocated(self%u_init).and.allocated(self%v_init)) then
      model=learn_dictionary(transpose(centered),k,self%alpha,self%max_iter,self%tol,trim(self%method), &
        1000,self%random_state,.false.,.false.,transpose(self%v_init),transpose(self%u_init))
    else
      model=learn_dictionary(transpose(centered),k,self%alpha,self%max_iter,self%tol,trim(self%method), &
        1000,self%random_state,.false.,.false.)
    end if
    self%components_=transpose(model%code)
    do j=1,k
      index=maxloc(abs(self%components_(j,:)),dim=1)
      if(self%components_(j,index)<0) self%components_(j,:)=-self%components_(j,:)
      norm=sqrt(sum(self%components_(j,:)**2))
      if(norm>0) self%components_(j,:)=self%components_(j,:)/norm
    end do
    self%n_iter_=model%n_iter; self%error_=model%errors; self%n_components_=k; self%n_features_in_=size(x,2)
  end subroutine
  function sparse_pca_fit_transform(self,x) result(z)
    class(sparse_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function sparse_projection(x,components,mean,ridge_alpha,name) result(z)
    real(dp), intent(in) :: x(:,:),components(:,:),mean(:),ridge_alpha
    character(*), intent(in) :: name
    real(dp), allocatable :: z(:,:),s(:),u(:,:),v(:,:),factor(:),centered(:,:)
    real(dp) :: cutoff
    integer :: k
    call check_x(x)
    call require(size(x,2)==size(components,2),name//' feature count mismatch')
    call require(ieee_is_finite(ridge_alpha),name//' ridge_alpha must be nonnegative')
    call require(ridge_alpha>=0,name//' ridge_alpha must be nonnegative')
    k=size(components,1); allocate(s(k),u(size(x,2),k),v(k,k),factor(k))
    call dense_svd(transpose(components),s,u,v)
    cutoff=epsilon(1._dp)*max(k,size(x,2))*maxval(s); factor=0
    if(ridge_alpha>0) then
      factor=s/(s*s+ridge_alpha)
    else
      where(s>cutoff) factor=1/max(s,tiny(1._dp))
    end if
    centered=x-spread(mean,1,size(x,1)); z=matmul(matmul(centered,u)*spread(factor,1,size(x,1)),transpose(v))
  end function
  function sparse_pca_transform(self,x) result(z)
    class(sparse_pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'SparsePCA is not fitted')
    z=sparse_projection(x,self%components_,self%mean_,self%ridge_alpha,'SparsePCA')
  end function
  function sparse_pca_inverse(self,z) result(x)
    class(sparse_pca), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'SparsePCA is not fitted'); call check_x(z)
    call require(size(z,2)==size(self%components_,1),'SparsePCA inverse component count mismatch')
    x=matmul(z,self%components_)+spread(self%mean_,1,size(z,1))
  end function
  subroutine mini_controls(self)
    class(mini_batch_dictionary_learning), intent(in) :: self
    call check_coding_parameters(self)
    call require(self%n_components>=0.and.self%n_components<=huge(1)-10, &
      'MiniBatchDictionaryLearning n_components must be nonnegative and within index capacity')
    call require(self%max_iter>=0,'MiniBatchDictionaryLearning max_iter must be nonnegative')
    call require(self%batch_size>=1,'MiniBatchDictionaryLearning batch_size must be positive')
    call require(self%random_state>=0,'MiniBatchDictionaryLearning seed must be nonnegative')
    call require(self%max_no_improvement>=-1,'MiniBatchDictionaryLearning patience must be nonnegative or -1 for none')
    call require(ieee_is_finite(self%alpha).and.ieee_is_finite(self%tol), &
      'MiniBatchDictionaryLearning controls must be finite')
    call require(self%alpha>=0.and.self%tol>=0,'MiniBatchDictionaryLearning alpha/tol must be nonnegative')
    call require(self%fit_algorithm=='lars'.or.self%fit_algorithm=='cd','unknown minibatch dictionary fit algorithm')
  end subroutine
  subroutine mini_initialize(self,x)
    class(mini_batch_dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:),left(:,:),right(:,:)
    integer :: k,p,r
    call require(.not.(self%positive_code.and.self%fit_algorithm=='lars'), &
      'positive minibatch dictionary codes require fit_algorithm cd')
    k=self%n_components; p=size(x,2)
    if(k==0) k=p
    call require(k<=huge(k)-10,'invalid randomized SVD component count')
    self%n_components_=k; self%n_features_in_=p; self%batch_size_=min(self%batch_size,size(x,1))
    self%fitted_algorithm=self%fit_algorithm; call self%rng%seed(self%random_state)
    self%components_=spread(spread(0._dp,1,k),2,p)
    if(allocated(self%dict_init)) then
      call require(size(self%dict_init,2)==p,'minibatch dictionary initializer feature count mismatch')
      call require(all(ieee_is_finite(self%dict_init)),'nonfinite minibatch dictionary initializer')
      r=min(k,size(self%dict_init,1)); self%components_(:r,:)=self%dict_init(:r,:)
    else
      call randomized_svd(x,k,self%rng,values,left,right); r=size(values)
      self%components_(:r,:)=right*spread(values,2,p)
    end if
    self%a_=spread(spread(0._dp,1,k),2,k); self%b_=spread(spread(0._dp,1,p),2,k)
    self%n_steps_=0; self%n_iter_=0; self%no_improvement_=0
    self%ewa_cost_=ieee_value(0._dp,ieee_quiet_nan); self%ewa_cost_min_=self%ewa_cost_
  end subroutine
  real(dp) function mini_step(self,x,step) result(cost)
    class(mini_batch_dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer(int64), intent(in) :: step
    real(dp), allocatable :: code(:,:)
    real(dp) :: theta,beta,batch
    type(legacy_random_state) :: rng
    code=sparse_encode(x,self%components_,algorithm='lasso_'//trim(self%fitted_algorithm),alpha=self%alpha, &
      max_iter=self%transform_max_iter,positive=self%positive_code)
    batch=real(size(x,1),dp)
    cost=dictionary_objective(x,code,self%components_,self%alpha)/batch
    if(step<int(size(x,1)-1,int64)) then
      theta=real(step+1,dp)*batch
    else
      theta=batch**2+real(step+1,dp)-batch
    end if
    beta=(theta+1-batch)/(theta+1)
    self%a_=beta*self%a_+matmul(transpose(code),code)/batch
    self%b_=beta*self%b_+matmul(transpose(x),code)/batch
    rng=self%rng; call update_dictionary(self%components_,x,code,self%a_,self%b_,rng,self%positive_dict); self%rng=rng
  end function
  logical function mini_converged(self,cost,old,n,actual_batch,step) result(stopped)
    class(mini_batch_dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: cost,old(:,:)
    integer, intent(in) :: n,actual_batch
    integer(int64), intent(in) :: step
    real(dp) :: weight,difference
    stopped=.false.
    if(real(step+1,dp)<=min(100._dp,real(n,dp)/actual_batch)) return
    if(ieee_is_nan(self%ewa_cost_)) then
      self%ewa_cost_=cost
    else
      weight=min(1._dp,real(actual_batch,dp)/(real(n,dp)+1))
      self%ewa_cost_=(1-weight)*self%ewa_cost_+weight*cost
    end if
    difference=sqrt(sum((self%components_-old)**2))/self%n_components_
    if(self%tol>0.and.difference<=self%tol) then
      stopped=.true.; return
    end if
    if(ieee_is_nan(self%ewa_cost_min_)) then
      self%ewa_cost_min_=self%ewa_cost_; self%no_improvement_=0
    else if(self%ewa_cost_<self%ewa_cost_min_) then
      self%ewa_cost_min_=self%ewa_cost_; self%no_improvement_=0
    else
      self%no_improvement_=self%no_improvement_+1
    end if
    if(self%max_no_improvement>=0) stopped=self%no_improvement_>=int(self%max_no_improvement,int64)
  end function
  subroutine mini_dictionary_fit(self,x)
    class(mini_batch_dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: training(:,:),old(:,:)
    integer :: permutation(size(x,1)),i,j,temp,n,batches,first,last
    integer(int64) :: total,step
    real(dp) :: cost
    call mini_controls(self); call check_x(x); call mini_initialize(self,x)
    n=size(x,1); permutation=[(i,i=1,n)]
    if(self%shuffle) then
      do i=n,2,-1
        j=self%rng%integer_index(i); temp=permutation(i); permutation(i)=permutation(j); permutation(j)=temp
      end do
    end if
    training=x(permutation,:); old=self%components_; first=1
    batches=(n-1)/self%batch_size_+1; total=int(self%max_iter,int64)*batches
    do step=0_int64,total-1
      last=first+min(self%batch_size_,n-first+1)-1
      cost=mini_step(self,training(first:last,:),step); self%n_steps_=step+1
      if(mini_converged(self,cost,old,n,last-first+1,step)) exit
      if(associated(self%callback))call self%callback(step,cost,self%components_,self%a_,self%b_)
      old=self%components_
      if(last==n) then
        first=1
      else
        first=last+1
      end if
    end do
    self%n_iter_=(self%n_steps_+batches-1)/batches
  end subroutine
  subroutine mini_dictionary_partial_fit(self,x)
    class(mini_batch_dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: cost
    integer :: k
    call mini_controls(self); call check_x(x)
    if(.not.allocated(self%components_)) then
      call mini_initialize(self,x)
    else
      call require(size(x,2)==self%n_features_in_,'MiniBatchDictionaryLearning feature count mismatch')
      k=self%n_components; if(k==0) k=self%n_features_in_
      call require(k==self%n_components_,'MiniBatchDictionaryLearning component count changed; use fit to reset')
      call require(.not.(self%positive_code.and.self%fitted_algorithm=='lars'), &
        'positive minibatch dictionary codes require fit_algorithm cd')
    end if
    call require(self%n_steps_<huge(self%n_steps_),'MiniBatchDictionaryLearning step counter capacity exceeded')
    cost=mini_step(self,x,self%n_steps_); self%n_steps_=self%n_steps_+1
  end subroutine
  function mini_dictionary_fit_transform(self,x) result(code)
    class(mini_batch_dictionary_learning), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: code(:,:)
    call self%fit(x); code=self%transform(x)
  end function
  function mini_dictionary_transform(self,x) result(code)
    class(mini_batch_dictionary_learning), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: code(:,:)
    call require(allocated(self%components_),'MiniBatchDictionaryLearning is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'MiniBatchDictionaryLearning feature count mismatch')
    code=apply_coding(self,x,self%components_,self%alpha)
  end function
  subroutine mini_sparse_pca_fit(self,x)
    class(mini_batch_sparse_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: centered(:,:),codes(:,:)
    real(dp) :: norm
    type(mini_batch_dictionary_learning) :: model
    integer :: k,j
    call check_x(x)
    call require(self%n_components>=0,'MiniBatchSparsePCA n_components must be nonnegative (zero means auto)')
    call require(self%max_iter>=0,'MiniBatchSparsePCA max_iter must be nonnegative')
    call require(self%batch_size>=1,'MiniBatchSparsePCA batch_size must be positive')
    call require(self%random_state>=0,'MiniBatchSparsePCA seed must be nonnegative')
    call require(self%max_no_improvement>=-1,'MiniBatchSparsePCA patience must be nonnegative or -1 for none')
    call require(ieee_is_finite(self%alpha).and.ieee_is_finite(self%tol),'MiniBatchSparsePCA controls must be finite')
    call require(self%alpha>=0.and.self%tol>=0,'MiniBatchSparsePCA alpha/tol must be nonnegative')
    call require(self%method=='lars'.or.self%method=='cd','unknown MiniBatchSparsePCA method')
    call require(ieee_is_finite(self%ridge_alpha),'MiniBatchSparsePCA ridge_alpha must be finite and nonnegative')
    call require(self%ridge_alpha>=0,'MiniBatchSparsePCA ridge_alpha must be finite and nonnegative')
    k=self%n_components; if(k==0) k=size(x,2)
    self%mean_=sum(x,dim=1)/size(x,1); centered=x-spread(self%mean_,1,size(x,1))
    model%n_components=k; model%alpha=self%alpha; model%max_iter=self%max_iter; model%batch_size=self%batch_size
    model%shuffle=self%shuffle; model%fit_algorithm=self%method; model%random_state=self%random_state
    model%transform_algorithm='lasso_'//trim(self%method); model%transform_alpha=self%alpha
    model%tol=self%tol; model%max_no_improvement=self%max_no_improvement
    call model%fit(transpose(centered)); codes=model%transform(transpose(centered)); self%components_=transpose(codes)
    do j=1,k
      norm=sqrt(sum(self%components_(j,:)**2))
      if(norm>0) self%components_(j,:)=self%components_(j,:)/norm
    end do
    self%n_iter_=model%n_iter_; self%n_steps_=model%n_steps_; self%n_components_=k; self%n_features_in_=size(x,2)
  end subroutine
  function mini_sparse_pca_fit_transform(self,x) result(z)
    class(mini_batch_sparse_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function mini_sparse_pca_transform(self,x) result(z)
    class(mini_batch_sparse_pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'MiniBatchSparsePCA is not fitted')
    z=sparse_projection(x,self%components_,self%mean_,self%ridge_alpha,'MiniBatchSparsePCA')
  end function
  function mini_sparse_pca_inverse(self,z) result(x)
    class(mini_batch_sparse_pca), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'MiniBatchSparsePCA is not fitted'); call check_x(z)
    call require(size(z,2)==self%n_components_,'MiniBatchSparsePCA inverse component count mismatch')
    x=matmul(z,self%components_)+spread(self%mean_,1,size(z,1))
  end function
end module
