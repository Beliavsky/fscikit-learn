! Native adaptation of sklearn 1.6.1 decomposition/_dict_learning.py and _sparse_pca.py.
! BSD-3-Clause upstream notices are retained in LICENSES and THIRD_PARTY_NOTICES.md.
module sklearn_dictionary
  use sklearn_common
  use sklearn_least_angle, only: lars_path,lars_path_result
  use sklearn_linear_model, only: orthogonal_matching_pursuit
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sparse_pca,dictionary_learning,sparse_coder,sparse_encode
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
  type :: learning_result
    real(dp), allocatable :: code(:,:),dictionary(:,:),errors(:)
    integer :: n_iter=0
  end type
contains
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
    call require(ieee_is_finite(penalty).and.penalty>=0,'sparse coding alpha must be finite and nonnegative')
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
  function learn_dictionary(y,k,alpha,limit,tolerance,method,inner_limit,seed,positive_dict, &
      positive_code,code_init,dict_init) result(model)
    real(dp), intent(in) :: y(:,:),alpha,tolerance
    integer, intent(in) :: k,limit,inner_limit,seed
    character(*), intent(in) :: method
    logical, intent(in) :: positive_dict,positive_code
    real(dp), intent(in), optional :: code_init(:,:),dict_init(:,:)
    type(learning_result) :: model
    type(legacy_random_state) :: rng
    real(dp), allocatable :: u(:,:),v(:,:),s(:),a(:,:),b(:,:),history(:),sample(:)
    real(dp) :: norm,scale,cost
    integer :: n,p,r,j,it,index
    call check_x(y); n=size(y,1); p=size(y,2)
    call require(k>=1,'dictionary component count must be positive')
    call require(limit>=0.and.limit<huge(limit).and.inner_limit>=0.and.inner_limit<huge(inner_limit), &
      'invalid dictionary iteration limit')
    call require(all(ieee_is_finite([alpha,tolerance])).and.alpha>=0.and.tolerance>=0, &
      'dictionary alpha and tolerance must be finite and nonnegative')
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
      do j=1,k
        if(a(j,j)>1.e-6_dp) then
          model%dictionary(j,:)=model%dictionary(j,:)+(b(:,j)-matmul(a(j,:),model%dictionary))/a(j,j)
        else
          index=rng%integer_index(n); sample=y(index,:)
          scale=sqrt(sum((sample-sum(sample)/p)**2)/p); if(scale==0) scale=1
          scale=.01_dp*scale
          do index=1,p
            sample(index)=sample(index)+scale*rng%normal()
          end do
          model%dictionary(j,:)=sample; model%code(:,j)=0
        end if
        if(positive_dict) model%dictionary(j,:)=max(model%dictionary(j,:),0._dp)
        norm=sqrt(sum(model%dictionary(j,:)**2)); model%dictionary(j,:)=model%dictionary(j,:)/max(norm,1._dp)
      end do
      cost=.5_dp*sum((y-matmul(model%code,model%dictionary))**2)+alpha*sum(abs(model%code))
      history(it)=cost; model%n_iter=it
      if(it>1) then
        if(history(it-1)-cost<tolerance*cost) exit
      end if
    end do
    model%errors=history(:model%n_iter)
  end function
  subroutine check_coding_parameters(self)
    class(coding_parameters), intent(in) :: self
    call require(valid_coding_algorithm(self%transform_algorithm),'unknown sparse coding algorithm')
    call require(self%transform_n_nonzero_coefs>=0,'sparse coding sparsity must be nonnegative (zero means auto)')
    call require(self%transform_max_iter>=0.and.self%transform_max_iter<huge(self%transform_max_iter), &
      'invalid sparse coding max_iter')
    call require(ieee_is_finite(self%transform_alpha).and. &
      (self%transform_alpha==-1.or.self%transform_alpha>=0),'transform alpha must be -1 (auto) or nonnegative')
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
      self%transform_max_iter,self%random_state,self%positive_dict,self%positive_code,self%code_init,self%dict_init)
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
    call require(ieee_is_finite(self%ridge_alpha).and.self%ridge_alpha>=0,'SparsePCA ridge_alpha must be nonnegative')
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
  function sparse_pca_transform(self,x) result(z)
    class(sparse_pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),s(:),u(:,:),v(:,:),factor(:),centered(:,:)
    real(dp) :: cutoff
    integer :: k
    call require(allocated(self%components_),'SparsePCA is not fitted'); call check_x(x)
    call require(size(x,2)==size(self%components_,2),'SparsePCA feature count mismatch')
    call require(ieee_is_finite(self%ridge_alpha).and.self%ridge_alpha>=0,'SparsePCA ridge_alpha must be nonnegative')
    k=size(self%components_,1); allocate(s(k),u(size(x,2),k),v(k,k),factor(k))
    call dense_svd(transpose(self%components_),s,u,v)
    cutoff=epsilon(1._dp)*max(k,size(x,2))*maxval(s); factor=0
    if(self%ridge_alpha>0) then
      factor=s/(s*s+self%ridge_alpha)
    else
      where(s>cutoff) factor=1/max(s,tiny(1._dp))
    end if
    centered=x-spread(self%mean_,1,size(x,1)); z=matmul(matmul(centered,u)*spread(factor,1,size(x,1)),transpose(v))
  end function
  function sparse_pca_inverse(self,z) result(x)
    class(sparse_pca), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'SparsePCA is not fitted'); call check_x(z)
    call require(size(z,2)==size(self%components_,1),'SparsePCA inverse component count mismatch')
    x=matmul(z,self%components_)+spread(self%mean_,1,size(z,1))
  end function
end module
