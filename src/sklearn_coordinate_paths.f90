! Elastic-net/Lasso paths: sklearn 1.6.1 _coordinate_descent.py/_cd_fast.pyx.
! Upstream BSD-3-Clause terms retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_coordinate_paths
  use sklearn_common,only:dp,require,check_x,check_xy,argsort
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_multitask,only:multi_task_elastic_net,multitask_path_step
  use sklearn_sparse,only:csr_matrix
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
  type,private::path_csc
    integer,allocatable::indptr(:),rows(:)
    real(dp),allocatable::data(:)
  end type
  interface enet_path
    module procedure enet_vector,enet_matrix,enet_sparse_vector
  end interface
  interface lasso_path
    module procedure lasso_vector,lasso_matrix,lasso_sparse_vector
  end interface
contains
  function enet_sparse_vector(x,y,options,alphas,coef_init,gram,xy,sample_weight,x_offset,x_scale)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    type(coordinate_path_options),optional,intent(in)::options
    real(dp),optional,intent(in)::alphas(:),coef_init(:),gram(:,:),xy(:),sample_weight(:),x_offset(:),x_scale(:)
    type(coordinate_path_result)::fit
    type(coordinate_path_options)::control
    type(legacy_random_state)::rng
    type(path_csc)::csc
    real(dp),allocatable::grid(:),coef(:),mean(:),weights(:),cross(:)
    real(dp)::maximum,gap,l1,l2
    integer,allocatable::order(:)
    integer::n,p,i,j,k
    if(present(options))control=options
    call x%validate();n=x%n_rows;p=x%n_cols
    call require(n>0.and.p>0.and.p<huge(1),'sparse path requires nonempty bounded dimensions')
    call require(size(y)==n,'sparse path target length mismatch')
    call require(all(ieee_is_finite(x%data)).and.all(ieee_is_finite(y)),'sparse path data must be finite')
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
    allocate(coef(p),mean(p),weights(n));coef=0;mean=0;weights=1
    if(present(coef_init))then
      call require(size(coef_init)==p,'path initial coefficient shape mismatch')
      call require(all(ieee_is_finite(coef_init)),'path initial coefficients must be finite')
      coef=coef_init
    end if
    if(present(sample_weight))then
      call require(size(sample_weight)==n,'sparse path weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'sparse path weights must be finite')
      weights=sample_weight
    end if
    if(present(x_offset))then
      call require(present(x_scale),'sparse path offsets require scales')
      call require(size(x_offset)==p.and.size(x_scale)==p,'sparse path offset/scale length mismatch')
      call require(all(ieee_is_finite(x_offset)).and.all(ieee_is_finite(x_scale)), &
        'sparse path offsets/scales must be finite')
      call require(all(x_scale/=0),'sparse path scales must be nonzero')
      do j=1,p
        if(abs(x_scale(j))<1)call require(abs(x_offset(j))<=huge(1._dp)*abs(x_scale(j)), &
          'sparse path means exceed working range')
      end do
      mean=x_offset/x_scale
      call require(all(ieee_is_finite(mean)),'sparse path means exceed working range')
    end if
    ! check_input validates Xy first, then sparse _pre_fit drops Gram and Xy.
    if(present(xy))call check_x(reshape(xy,[size(xy),1]))
    csc=csr_columns(x)
    if(present(alphas))then
      call require(all(ieee_is_finite(alphas)),'path alphas must be finite')
      call require(all(alphas>=0),'path alphas must be nonnegative')
      grid=alphas
      if(size(grid)>1)then
        order=argsort(grid);grid=grid(order(size(order):1:-1))
      end if
    else
      call require(control%l1_ratio>0,'automatic path grid requires positive l1_ratio')
      allocate(cross(p));cross=0
      do j=1,p
        do k=csc%indptr(j),csc%indptr(j+1)-1
          cross(j)=cross(j)+csc%data(k)*y(csc%rows(k))
        end do
      end do
      allocate(grid(control%n_alphas))
      ! Public enet_path's sparse alpha grid ignores weights and X_offset.
      maximum=maxval(sqrt(cross**2))/(n*control%l1_ratio)
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
        grid(1)=maximum
        if(size(grid)>1)grid(size(grid))=maximum*control%eps
      end if
    end if
    fit%alphas=grid;fit%backend='sparse';fit%dual_tolerance=control%tol*dot_product(y,weights*y)/n
    allocate(fit%coefficients(1,p,size(grid)),fit%dual_gaps(size(grid)), &
      fit%n_iter(size(grid)),fit%converged(size(grid)))
    call rng%seed(control%random_state)
    do i=1,size(grid)
      l1=grid(i)*control%l1_ratio*n;l2=grid(i)*(1-control%l1_ratio)*n
      call sparse_step(csc,n,y,coef,weights,present(sample_weight),mean,l1,l2,control,rng, &
        gap,fit%n_iter(i),fit%converged(i))
      fit%coefficients(1,:,i)=coef;fit%dual_gaps(i)=gap
    end do
  end function
  function lasso_sparse_vector(x,y,options,alphas,coef_init,gram,xy,sample_weight,x_offset,x_scale)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    type(coordinate_path_options),optional,intent(in)::options
    real(dp),optional,intent(in)::alphas(:),coef_init(:),gram(:,:),xy(:),sample_weight(:),x_offset(:),x_scale(:)
    type(coordinate_path_result)::fit
    type(coordinate_path_options)::control
    if(present(options))control=options
    control%l1_ratio=1
    fit=enet_sparse_vector(x,y,control,alphas,coef_init,gram,xy,sample_weight,x_offset,x_scale)
  end function
  function csr_columns(x)result(csc)
    type(csr_matrix),intent(in)::x
    type(path_csc)::csc
    integer,allocatable::counts(:),next(:)
    integer::i,j,k,q
    allocate(counts(x%n_cols),csc%indptr(x%n_cols+1),csc%rows(size(x%data)),csc%data(size(x%data)))
    counts=0
    do k=1,size(x%data)
      j=x%indices(k);counts(j)=counts(j)+1
    end do
    csc%indptr(1)=1
    do j=1,x%n_cols
      csc%indptr(j+1)=csc%indptr(j)+counts(j)
    end do
    next=csc%indptr(:x%n_cols)
    do i=1,x%n_rows
      do k=x%indptr(i),x%indptr(i+1)-1
        j=x%indices(k);q=next(j);next(j)=q+1
        csc%rows(q)=i;csc%data(q)=x%data(k)
      end do
    end do
  end function
  subroutine sparse_step(csc,n,y,w,weights,weighted,mean,l1,l2,control,rng,gap,iterations,converged)
    type(path_csc),intent(in)::csc
    integer,intent(in)::n
    real(dp),intent(in)::y(:),weights(:),mean(:),l1,l2
    real(dp),intent(inout)::w(:)
    logical,intent(in)::weighted
    type(coordinate_path_options),intent(in)::control
    type(legacy_random_state),intent(inout)::rng
    real(dp),intent(out)::gap
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp),allocatable::residual(:),yw(:),norms(:),gradient(:)
    real(dp)::total,old,value,change,largest,rsum,threshold,rnorm2,constant,dualnorm,wnorm2,weight
    integer::p,j,k,i,row,it,q
    integer(int64)::state,unstored_count
    logical::center,small
    p=size(w);state=int(rng%integer_index(2147483647)-1,int64)
    yw=y;if(weighted)yw=weights*y
    residual=yw;allocate(norms(p),gradient(p));norms=0;center=any(mean/=0)
    do j=1,p
      total=0
      do k=csc%indptr(j),csc%indptr(j+1)-1
        row=csc%rows(k);value=csc%data(k)
        if(weighted)then
          weight=weights(row)
          total=total+(weight*(value-mean(j))**2-weight*mean(j)**2)
          residual(row)=residual(row)-weight*value*w(j)
        else
          total=total+(value-mean(j))**2
          residual(row)=residual(row)-value*w(j)
        end if
      end do
      if(weighted)then
        if(center)then
          do i=1,n
            total=total+weights(i)*mean(j)**2
            residual(i)=residual(i)+weights(i)*mean(j)*w(j)
          end do
        end if
        norms(j)=total
      else
        ! Cython uses unsigned 32-bit counters, including with duplicate rows.
        unstored_count=modulo(int(n,int64)-int(csc%indptr(j+1),int64)+int(csc%indptr(j),int64),4294967296_int64)
        norms(j)=total+real(unstored_count,dp)*mean(j)**2
        if(center)residual=residual+mean(j)*w(j)
      end if
    end do
    gap=control%tol+1;threshold=control%tol*dot_product(y,yw);iterations=1;converged=.false.
    do it=1,control%max_iter
      change=0;largest=0
      do q=1,p
        j=q;if(control%selection=='random')j=random_coordinate(state,p)
        if(norms(j)==0)cycle
        old=w(j)
        if(old/=0)then
          do k=csc%indptr(j),csc%indptr(j+1)-1
            row=csc%rows(k)
            if(weighted)then
              residual(row)=residual(row)+weights(row)*csc%data(k)*old
            else
              residual(row)=residual(row)+csc%data(k)*old
            end if
          end do
          if(center)then
            if(weighted)then
              residual=residual-weights*mean(j)*old
            else
              residual=residual-mean(j)*old
            end if
          end if
        end if
        value=0
        do k=csc%indptr(j),csc%indptr(j+1)-1
          value=value+residual(csc%rows(k))*csc%data(k)
        end do
        if(center)then
          rsum=sum(residual);value=value-rsum*mean(j)
        end if
        call require(norms(j)+l2/=0,'zero sparse path coordinate denominator')
        if(control%positive.and.value<0)then
          w(j)=0
        else
          w(j)=sign(max(abs(value)-l1,0._dp),value)/(norms(j)+l2)
        end if
        if(w(j)/=0)then
          do k=csc%indptr(j),csc%indptr(j+1)-1
            row=csc%rows(k)
            if(weighted)then
              residual(row)=residual(row)-weights(row)*csc%data(k)*w(j)
            else
              residual(row)=residual(row)-csc%data(k)*w(j)
            end if
          end do
          if(center)then
            if(weighted)then
              residual=residual+weights*mean(j)*w(j)
            else
              residual=residual+mean(j)*w(j)
            end if
          end if
        end if
        change=max(change,abs(w(j)-old));largest=max(largest,abs(w(j)))
      end do
      small=largest==0
      if(largest/=0)small=change/largest<control%tol
      if(small.or.it==control%max_iter)then
        rsum=0;if(center)rsum=sum(residual)
        do j=1,p
          gradient(j)=0
          do k=csc%indptr(j),csc%indptr(j+1)-1
            gradient(j)=gradient(j)+csc%data(k)*residual(csc%rows(k))
          end do
          if(center)gradient(j)=gradient(j)-mean(j)*rsum
          gradient(j)=gradient(j)-l2*w(j)
        end do
        dualnorm=maxval(abs(gradient));if(control%positive)dualnorm=maxval(gradient)
        if(weighted)then
          rnorm2=0
          do i=1,n
            if(weights(i)/=0)rnorm2=rnorm2+residual(i)**2/weights(i)
          end do
        else
          rnorm2=dot_product(residual,residual)
        end if
        constant=1;gap=rnorm2;wnorm2=dot_product(w,w)
        if(dualnorm>l1)then
          constant=l1/dualnorm;gap=.5_dp*(rnorm2+rnorm2*constant**2)
        end if
        gap=gap+l1*sum(abs(w))-constant*dot_product(residual,y)+.5_dp*l2*(1+constant**2)*wnorm2
        if(gap<threshold)then
          converged=.true.;exit
        end if
      end if
    end do
    if(control%max_iter>0)iterations=min(it,control%max_iter)
    gap=gap/n
  end subroutine
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
