! Cholesky OMP and Gram OMP adapted from sklearn 1.6.1 linear_model/_omp.py.
! BSD-3-Clause attribution is retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_omp_paths
  use sklearn_common,only:dp,require,check_x,check_xy,solve_lower,solve_upper
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public::orthogonal_mp,orthogonal_mp_gram,omp_result
  type::omp_result
    real(dp),allocatable::coefficients(:,:),path(:,:,:)
    integer,allocatable::n_iter(:)
    logical,allocatable::premature(:)
    logical::gram_used=.false.
  end type
  interface orthogonal_mp
    module procedure mp_vector,mp_matrix,mp_vector_auto,mp_matrix_auto
  end interface
  interface orthogonal_mp_gram
    module procedure gram_vector,gram_matrix
  end interface
contains
  subroutine controls(p,default_min,n_nonzero_coefs,tol,limit)
    integer,intent(in)::p,default_min
    integer,optional,intent(in)::n_nonzero_coefs
    real(dp),optional,intent(in)::tol
    integer,intent(out)::limit
    limit=max(p/10,default_min)
    if(present(n_nonzero_coefs))then
      call require(n_nonzero_coefs>0,'OMP number of atoms must be positive')
      limit=n_nonzero_coefs
    end if
    if(present(tol))then
      call require(ieee_is_finite(tol),'OMP tolerance must be finite')
      call require(tol>=0,'OMP tolerance must be nonnegative')
      limit=p
    else
      call require(limit>0,'OMP number of atoms must be positive')
      call require(limit<=p,'OMP number of atoms exceeds features')
    end if
  end subroutine
  function mp_vector(x,y,n_nonzero_coefs,tol,precompute,return_path)result(fit)
    real(dp),intent(in)::x(:,:),y(:)
    integer,optional,intent(in)::n_nonzero_coefs
    real(dp),optional,intent(in)::tol
    logical,optional,intent(in)::precompute,return_path
    type(omp_result)::fit
    fit=mp_matrix(x,reshape(y,[size(y),1]),n_nonzero_coefs,tol,precompute,return_path)
  end function
  function mp_matrix(x,y,n_nonzero_coefs,tol,precompute,return_path)result(fit)
    real(dp),intent(in)::x(:,:),y(:,:)
    integer,optional,intent(in)::n_nonzero_coefs
    real(dp),optional,intent(in)::tol
    logical,optional,intent(in)::precompute,return_path
    type(omp_result)::fit
    logical::use_gram,keep
    integer::limit,p,t,k
    real(dp),allocatable::history(:,:),all_paths(:,:,:),gram(:,:),xy(:,:)
    call check_xy(x,size(y,1));call check_x(y)
    p=size(x,2);t=size(y,2)
    call controls(p,1,n_nonzero_coefs,tol,limit)
    use_gram=.false.;if(present(precompute))use_gram=precompute
    keep=.false.;if(present(return_path))keep=return_path
    if(use_gram)then
      gram=matmul(transpose(x),x);xy=matmul(transpose(x),y)
      if(present(tol))then
        fit=gram_matrix(gram,xy,limit,tol,sum(y**2,dim=1),keep)
      else
        fit=gram_matrix(gram,xy,limit,return_path=keep)
      end if
      return
    end if
    allocate(fit%coefficients(p,t),fit%n_iter(t),fit%premature(t))
    fit%coefficients=0
    if(keep)allocate(all_paths(p,t,limit))
    if(keep)all_paths=0
    do k=1,t
      call direct_path(x,y(:,k),limit,tol,keep,fit%coefficients(:,k),history,fit%n_iter(k),fit%premature(k))
      if(keep)all_paths(:,k,:fit%n_iter(k))=history
    end do
    if(keep)fit%path=all_paths(:,:,:maxval(fit%n_iter))
  end function
  function mp_vector_auto(x,y,precompute,n_nonzero_coefs,tol,return_path)result(fit)
    real(dp),intent(in)::x(:,:),y(:)
    character(*),intent(in)::precompute
    integer,optional,intent(in)::n_nonzero_coefs
    real(dp),optional,intent(in)::tol
    logical,optional,intent(in)::return_path
    type(omp_result)::fit
    call require(precompute=='auto','unknown OMP precompute mode')
    fit=mp_vector(x,y,n_nonzero_coefs,tol,size(x,1)>size(x,2),return_path)
  end function
  function mp_matrix_auto(x,y,precompute,n_nonzero_coefs,tol,return_path)result(fit)
    real(dp),intent(in)::x(:,:),y(:,:)
    character(*),intent(in)::precompute
    integer,optional,intent(in)::n_nonzero_coefs
    real(dp),optional,intent(in)::tol
    logical,optional,intent(in)::return_path
    type(omp_result)::fit
    call require(precompute=='auto','unknown OMP precompute mode')
    fit=mp_matrix(x,y,n_nonzero_coefs,tol,size(x,1)>size(x,2),return_path)
  end function
  function gram_vector(gram,xy,n_nonzero_coefs,tol,norms_squared,return_path)result(fit)
    real(dp),intent(in)::gram(:,:),xy(:)
    integer,optional,intent(in)::n_nonzero_coefs
    real(dp),optional,intent(in)::tol,norms_squared
    logical,optional,intent(in)::return_path
    type(omp_result)::fit
    if(present(norms_squared))then
      fit=gram_matrix(gram,reshape(xy,[size(xy),1]),n_nonzero_coefs,tol,[norms_squared],return_path)
    else
      fit=gram_matrix(gram,reshape(xy,[size(xy),1]),n_nonzero_coefs,tol,return_path=return_path)
    end if
  end function
  function gram_matrix(gram,xy,n_nonzero_coefs,tol,norms_squared,return_path)result(fit)
    real(dp),intent(in)::gram(:,:),xy(:,:)
    integer,optional,intent(in)::n_nonzero_coefs
    real(dp),optional,intent(in)::tol,norms_squared(:)
    logical,optional,intent(in)::return_path
    type(omp_result)::fit
    real(dp),allocatable::history(:,:),all_paths(:,:,:)
    real(dp)::norm
    integer::p,t,k,limit
    logical::keep
    call check_x(gram);call check_x(xy)
    p=size(gram,1);t=size(xy,2)
    call require(size(gram,2)==p,'OMP Gram must be square')
    call require(size(xy,1)==p,'OMP Gram and correlations feature counts differ')
    call controls(p,0,n_nonzero_coefs,tol,limit)
    if(present(tol))then
      call require(present(norms_squared),'Gram OMP requires squared target norms with tolerance')
      call require(size(norms_squared)==t,'OMP target norm count mismatch')
      call require(all(ieee_is_finite(norms_squared)),'OMP target norms must be finite')
    end if
    keep=.false.;if(present(return_path))keep=return_path
    fit%gram_used=.true.
    allocate(fit%coefficients(p,t),fit%n_iter(t),fit%premature(t));fit%coefficients=0
    if(keep)allocate(all_paths(p,t,limit))
    if(keep)all_paths=0
    do k=1,t
      norm=0;if(present(tol))norm=norms_squared(k)
      call gram_path(gram,xy(:,k),limit,tol,norm,keep,fit%coefficients(:,k),history,fit%n_iter(k),fit%premature(k))
      if(keep)all_paths(:,k,:fit%n_iter(k))=history
    end do
    if(keep)fit%path=all_paths(:,:,:maxval(fit%n_iter))
  end function
  real(dp) function vector_norm(x)result(value)
    real(dp),intent(in)::x(:)
    real(dp)::scale
    value=0
    if(size(x)==0)return
    scale=maxval(abs(x))
    if(scale==0)return
    if(scale>=sqrt(tiny(1._dp)).and.scale<=sqrt(huge(1._dp)/size(x)))then
      value=sqrt(sum(x**2))
    else
      value=scale*sqrt(sum((x/scale)**2))
    end if
  end function
  subroutine direct_path(x,y,limit,tol,keep,coef,path,steps,premature)
    real(dp),intent(in)::x(:,:),y(:)
    integer,intent(in)::limit
    real(dp),optional,intent(in)::tol
    logical,intent(in)::keep
    real(dp),intent(out)::coef(:)
    real(dp),allocatable,intent(out)::path(:,:)
    integer,intent(out)::steps
    logical,intent(out)::premature
    real(dp),allocatable::a(:,:),alpha(:),residual(:),l(:,:),row(:),gamma(:),column(:),history(:,:)
    real(dp)::diagonal,value
    integer::p,k,j,swap_index
    integer,allocatable::order(:)
    p=size(x,2);a=x;alpha=matmul(transpose(a),y);residual=y
    order=[(j,j=1,p)];allocate(l(limit,limit));l=0;steps=0;coef=0;premature=.false.
    if(keep)allocate(history(p,limit))
    if(keep)history=0
    do k=1,limit
      j=maxloc(abs(matmul(transpose(a),residual)),dim=1)
      if(j<k.or.abs(alpha(j))<sqrt(epsilon(1._dp)))then
        premature=.true.;exit
      end if
      if(k>1)then
        row=solve_lower(l(:k-1,:k-1),matmul(transpose(a(:,:k-1)),a(:,j)))
        diagonal=vector_norm(a(:,j))**2-vector_norm(row)**2
        if(diagonal<=epsilon(1._dp))then
          premature=.true.;exit
        end if
        l(k,:k-1)=row;l(k,k)=sqrt(diagonal)
      else
        l(k,k)=vector_norm(a(:,j))
        call require(l(k,k)>0,'OMP selected atom has nonpositive norm')
      end if
      column=a(:,k);a(:,k)=a(:,j);a(:,j)=column
      value=alpha(k);alpha(k)=alpha(j);alpha(j)=value
      swap_index=order(k);order(k)=order(j);order(j)=swap_index
      gamma=solve_upper(transpose(l(:k,:k)),solve_lower(l(:k,:k),alpha(:k)))
      coef=0;coef(order(:k))=gamma;steps=k
      if(keep)history(:,k)=coef
      residual=y-matmul(a(:,:k),gamma)
      if(present(tol))then
        if(vector_norm(residual)**2<=tol)exit
      end if
    end do
    if(keep)path=history(:,:steps)
  end subroutine
  subroutine gram_path(gram,xy,limit,tol,norm,keep,coef,path,steps,premature)
    real(dp),intent(in)::gram(:,:),xy(:),norm
    integer,intent(in)::limit
    real(dp),optional,intent(in)::tol
    logical,intent(in)::keep
    real(dp),intent(out)::coef(:)
    real(dp),allocatable,intent(out)::path(:,:)
    integer,intent(out)::steps
    logical,intent(out)::premature
    real(dp),allocatable::g(:,:),rhs(:),alpha(:),l(:,:),row(:),gamma(:),column(:),beta(:),history(:,:)
    real(dp)::diagonal,value,remaining,delta
    integer::p,k,j,swap_index
    integer,allocatable::order(:)
    p=size(xy);g=gram;rhs=xy;alpha=xy;remaining=norm;delta=0
    order=[(j,j=1,p)];allocate(l(limit,limit));l=0;steps=0;coef=0;premature=.false.
    if(keep)allocate(history(p,limit))
    if(keep)history=0
    do k=1,limit
      j=maxloc(abs(alpha),dim=1)
      if(j<k.or.abs(alpha(j))<sqrt(epsilon(1._dp)))then
        premature=.true.;exit
      end if
      if(k>1)then
        row=solve_lower(l(:k-1,:k-1),g(j,:k-1))
        diagonal=g(j,j)-vector_norm(row)**2
        if(diagonal<=epsilon(1._dp))then
          premature=.true.;exit
        end if
        l(k,:k-1)=row;l(k,k)=sqrt(diagonal)
      else
        call require(g(j,j)>0,'OMP selected Gram atom has nonpositive diagonal')
        l(k,k)=sqrt(g(j,j))
      end if
      column=g(k,:);g(k,:)=g(j,:);g(j,:)=column
      column=g(:,k);g(:,k)=g(:,j);g(:,j)=column
      value=rhs(k);rhs(k)=rhs(j);rhs(j)=value
      swap_index=order(k);order(k)=order(j);order(j)=swap_index
      gamma=solve_upper(transpose(l(:k,:k)),solve_lower(l(:k,:k),rhs(:k)))
      coef=0;coef(order(:k))=gamma;steps=k
      if(keep)history(:,k)=coef
      beta=matmul(g(:,:k),gamma);alpha=rhs-beta
      if(present(tol))then
        remaining=remaining+delta;delta=dot_product(gamma,beta(:k));remaining=remaining-delta
        if(abs(remaining)<=tol)exit
      end if
    end do
    ! Source may make one final failed selection after exhausting all atoms.
    if(present(tol))then
      if(steps==limit.and.abs(remaining)>tol)premature=.true.
    end if
    if(keep)path=history(:,:steps)
  end subroutine
end module
