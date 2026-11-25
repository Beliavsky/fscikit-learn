! Elastic-net/Lasso paths: sklearn 1.6.1 _coordinate_descent.py/_cd_fast.pyx.
! Upstream BSD-3-Clause terms retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_coordinate_paths
  use sklearn_common,only:dp,require,check_x,check_xy,argsort
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_multitask,only:multi_task_elastic_net,multitask_path_step
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public::enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  type::coordinate_path_options
    real(dp)::l1_ratio=.5_dp,eps=1.e-3_dp,tol=1.e-4_dp
    integer::n_alphas=100,max_iter=1000,random_state=0
    logical::positive=.false.
    character(16)::selection='cyclic',precompute='auto'
  end type
  type::coordinate_path_result
    real(dp),allocatable::alphas(:),coefficients(:,:,:),dual_gaps(:)
    integer,allocatable::n_iter(:)
    logical,allocatable::converged(:)
    real(dp)::dual_tolerance=0
    character(16)::backend=''
  end type
  interface enet_path
    module procedure enet_vector,enet_matrix
  end interface
  interface lasso_path
    module procedure lasso_vector,lasso_matrix
  end interface
contains
  function enet_vector(x,y,options,alphas,coef_init,gram,xy)result(fit)
    real(dp),intent(in)::x(:,:),y(:)
    type(coordinate_path_options),optional,intent(in)::options
    real(dp),optional,intent(in)::alphas(:),coef_init(:),gram(:,:),xy(:)
    type(coordinate_path_result)::fit
    real(dp),allocatable::initial(:,:),correlation(:,:)
    if(present(coef_init))initial=reshape(coef_init,[1,size(coef_init)])
    if(present(xy))correlation=reshape(xy,[size(xy),1])
    fit=path_core(x,reshape(y,[size(y),1]),.false.,options,alphas,initial,gram,correlation)
  end function
  function enet_matrix(x,y,options,alphas,coef_init,gram,xy)result(fit)
    real(dp),intent(in)::x(:,:),y(:,:)
    type(coordinate_path_options),optional,intent(in)::options
    real(dp),optional,intent(in)::alphas(:),coef_init(:,:),gram(:,:),xy(:,:)
    type(coordinate_path_result)::fit
    fit=path_core(x,y,.true.,options,alphas,coef_init,gram,xy)
  end function
  function lasso_vector(x,y,options,alphas,coef_init,gram,xy)result(fit)
    real(dp),intent(in)::x(:,:),y(:)
    type(coordinate_path_options),optional,intent(in)::options
    real(dp),optional,intent(in)::alphas(:),coef_init(:),gram(:,:),xy(:)
    type(coordinate_path_options)::control
    type(coordinate_path_result)::fit
    if(present(options))control=options
    control%l1_ratio=1
    fit=enet_vector(x,y,control,alphas,coef_init,gram,xy)
  end function
  function lasso_matrix(x,y,options,alphas,coef_init,gram,xy)result(fit)
    real(dp),intent(in)::x(:,:),y(:,:)
    type(coordinate_path_options),optional,intent(in)::options
    real(dp),optional,intent(in)::alphas(:),coef_init(:,:),gram(:,:),xy(:,:)
    type(coordinate_path_options)::control
    type(coordinate_path_result)::fit
    if(present(options))control=options
    control%l1_ratio=1
    fit=enet_matrix(x,y,control,alphas,coef_init,gram,xy)
  end function
  function path_core(x,y,multi,options,alphas,coef_init,gram,xy)result(fit)
    real(dp),intent(in)::x(:,:),y(:,:)
    logical,intent(in)::multi
    type(coordinate_path_options),optional,intent(in)::options
    real(dp),optional,intent(in)::alphas(:),coef_init(:,:),gram(:,:),xy(:,:)
    type(coordinate_path_result)::fit
    type(coordinate_path_options)::control
    type(multi_task_elastic_net)::model
    type(legacy_random_state)::rng
    real(dp),allocatable::g(:,:),cross(:,:),coef(:,:),grid(:)
    real(dp)::maximum,expected,actual,gap,l1,l2
    integer::n,p,t,i,j,k,na,unused
    integer,allocatable::order(:)
    logical::precompute
    if(present(options))control=options
    call check_xy(x,size(y,1));call check_x(y)
    n=size(x,1);p=size(x,2);t=size(y,2)
    call require(ieee_is_finite(control%l1_ratio),'nonfinite path l1_ratio')
    call require(control%l1_ratio>=0.and.control%l1_ratio<=1,'path l1_ratio must be in [0,1]')
    call require(ieee_is_finite(control%tol),'nonfinite path tolerance')
    call require(control%tol>=0.and.control%max_iter>=0,'invalid path iteration controls')
    call require(control%random_state>=0,'path random_state must be nonnegative')
    call require(ieee_is_finite(control%eps),'nonfinite path grid eps')
    call require(control%eps>0.and.control%n_alphas>=1,'invalid path grid controls')
    call require(control%selection=='cyclic'.or.control%selection=='random','unknown path coordinate selection')
    call require(control%precompute=='auto'.or.control%precompute=='true'.or.control%precompute=='false', &
      'unknown path precompute mode')
    call require(.not.(multi.and.control%positive),'positive path requires vector targets')
    if(present(coef_init))then
      call require(all(shape(coef_init)==[t,p]),'path initial coefficient shape mismatch')
      call require(all(ieee_is_finite(coef_init)),'path initial coefficients must be finite')
      coef=coef_init
    else
      allocate(coef(t,p));coef=0
    end if
    if(present(xy))then
      call check_x(xy)
    end if
    precompute=control%precompute=='true'.or.(control%precompute=='auto'.and.n>p)
    if(present(gram))then
      call require(all(shape(gram)==[p,p]),'path Gram shape mismatch')
      call require(all(ieee_is_finite(gram)),'path Gram must be finite')
      j=p/2+1;k=min(j+1,p)
      expected=dot_product(x(:,j),x(:,k));actual=gram(j,k)
      call require(abs(expected-actual)<=1.e-5_dp+1.e-7_dp*abs(actual),'path Gram failed sampled validation')
      g=gram;precompute=.true.
    else if(precompute.and..not.multi)then
      g=matmul(transpose(x),x)
    end if
    ! _pre_fit drops caller Xy when precompute is false, including for grids.
    if(precompute.and.present(xy))then
      call require(all(shape(xy)==[p,t]),'path correlation shape mismatch')
      cross=xy
    else
      cross=matmul(transpose(x),y)
    end if
    if(present(alphas))then
      call require(all(ieee_is_finite(alphas)),'path alphas must be finite')
      call require(all(alphas>=0),'path alphas must be nonnegative')
      grid=alphas
      if(size(grid)>1)then
        order=argsort(grid);grid=grid(order(size(order):1:-1))
      end if
    else
      call require(control%l1_ratio>0,'automatic path grid requires positive l1_ratio')
      allocate(grid(control%n_alphas))
      maximum=maxval(sqrt(sum(cross**2,dim=2)))/(n*control%l1_ratio)
      if(maximum<=1.e-15_dp)then
        grid=1.e-15_dp
      else
        do i=1,size(grid)
          if(size(grid)==1)then
            grid(i)=maximum
          else
            grid(i)=10._dp**(log10(maximum)+(i-1)*(log10(maximum*control%eps)-log10(maximum))/(size(grid)-1))
          end if
        end do
        if(size(grid)>0)grid(1)=maximum
        if(size(grid)>1)grid(size(grid))=maximum*control%eps
      end if
    end if
    fit%alphas=grid;na=size(grid)
    allocate(fit%coefficients(t,p,na),fit%dual_gaps(na),fit%n_iter(na),fit%converged(na))
    fit%backend='direct'
    if(precompute)fit%backend='gram'
    if(multi)fit%backend='multitask'
    fit%dual_tolerance=control%tol*sum(y*y)/n
    call rng%seed(control%random_state)
    model%fit_intercept=.false.;model%warm_start=.true.;model%l1_ratio=control%l1_ratio
    model%tol=control%tol;model%max_iter=control%max_iter;model%selection=control%selection
    do i=1,na
      l1=grid(i)*control%l1_ratio*n;l2=grid(i)*(1-control%l1_ratio)*n
      if(multi)then
        if(control%max_iter==0)then
          unused=rng%integer_index(2147483647)
          gap=(control%tol+1)/n;fit%n_iter(i)=1;fit%converged(i)=.false.
        else
          model%alpha=grid(i);model%coef_=coef
          call multitask_path_step(model,x,y,rng)
          coef=model%coef_;gap=model%dual_gap_;fit%n_iter(i)=model%n_iter_;fit%converged(i)=model%converged_
        end if
      else
        if(precompute)then
          call scalar_step(x,y(:,1),coef(1,:),l1,l2,control,rng,gap,fit%n_iter(i),fit%converged(i),g,cross(:,1))
        else
          call scalar_step(x,y(:,1),coef(1,:),l1,l2,control,rng,gap,fit%n_iter(i),fit%converged(i))
        end if
      end if
      fit%coefficients(:,:,i)=coef;fit%dual_gaps(i)=gap
    end do
  end function
  integer function random_coordinate(state,p)result(j)
    integer(int64),intent(inout)::state
    integer,intent(in)::p
    integer(int64),parameter::mask32=int(z'FFFFFFFF',int64),mask31=int(z'7FFFFFFF',int64)
    if(state==0)state=1
    state=iand(ieor(state,ishft(state,13)),mask32)
    state=ieor(state,ishft(state,-17));state=iand(ieor(state,ishft(state,5)),mask32)
    j=int(modulo(iand(state,mask31),int(p,int64)))+1
  end function
  subroutine scalar_step(x,y,w,l1,l2,control,rng,gap,iterations,converged,gram,xy)
    real(dp),intent(in)::x(:,:),y(:),l1,l2
    real(dp),intent(inout)::w(:)
    type(coordinate_path_options),intent(in)::control
    type(legacy_random_state),intent(inout)::rng
    real(dp),intent(out)::gap
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp),optional,intent(in)::gram(:,:),xy(:)
    real(dp),allocatable::residual(:),h(:),norms(:),gradient(:)
    real(dp)::old,value,change,largest,dualnorm,rnorm2,wnorm2,constant,qw,ynorm2,threshold,total
    integer::n,p,it,j,k
    integer(int64)::state
    logical::small
    n=size(x,1);p=size(x,2);state=int(rng%integer_index(2147483647)-1,int64)
    ynorm2=dot_product(y,y);threshold=control%tol*ynorm2
    gap=control%tol+1;iterations=1;converged=.false.
    if(present(gram))then
      h=matmul(gram,w);norms=[(gram(j,j),j=1,p)]
    else
      residual=y-matmul(x,w);norms=sum(x*x,dim=1)
    end if
    do it=1,control%max_iter
      change=0;largest=0
      do k=1,p
        j=k;if(control%selection=='random')j=random_coordinate(state,p)
        if(norms(j)==0)cycle
        old=w(j)
        if(present(gram))then
          if(old/=0)h=h-old*gram(j,:)
          value=xy(j)-h(j)
        else
          if(old/=0)residual=residual+old*x(:,j)
          value=dot_product(x(:,j),residual)
        end if
        call require(norms(j)+l2>0,'nonpositive path coordinate denominator')
        if(control%positive.and.value<0)then
          w(j)=0
        else
          w(j)=sign(max(abs(value)-l1,0._dp),value)/(norms(j)+l2)
        end if
        if(present(gram))then
          if(w(j)/=0)h=h+w(j)*gram(j,:)
        else
          if(w(j)/=0)residual=residual-w(j)*x(:,j)
        end if
        change=max(change,abs(w(j)-old));largest=max(largest,abs(w(j)))
      end do
      small=largest==0
      if(largest/=0)small=change/largest<control%tol
      if(small.or.it==control%max_iter)then
        if(present(gram))then
          qw=dot_product(w,xy);gradient=xy-h-l2*w
          total=0
          do j=1,p
            total=total+w(j)*h(j)
          end do
          rnorm2=ynorm2+total-2*qw
        else
          gradient=matmul(transpose(x),residual)-l2*w;rnorm2=dot_product(residual,residual)
        end if
        dualnorm=maxval(abs(gradient));if(control%positive)dualnorm=maxval(gradient)
        wnorm2=dot_product(w,w);constant=1;gap=rnorm2
        if(dualnorm>l1)then
          constant=l1/dualnorm;gap=.5_dp*(rnorm2+rnorm2*constant**2)
        end if
        if(present(gram))then
          gap=gap+l1*sum(abs(w))-constant*ynorm2+constant*qw+.5_dp*l2*(1+constant**2)*wnorm2
        else
          gap=gap+l1*sum(abs(w))-constant*dot_product(residual,y)+.5_dp*l2*(1+constant**2)*wnorm2
        end if
        if(gap<threshold)then
          converged=.true.;exit
        end if
      end if
    end do
    if(control%max_iter>0)iterations=min(it,control%max_iter)
    gap=gap/n
  end subroutine
end module
