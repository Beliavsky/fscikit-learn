program reference_coordinate_paths_driver
  use sklearn_coordinate_paths,only:enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  use sklearn_common,only:dp,require
  implicit none
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit
  real(dp),allocatable::x(:,:),y(:,:),grid(:),initial(:,:),gram(:,:),xy(:,:),xb(:,:),yb(:,:)
  real(dp),allocatable::ib(:,:),gb(:,:),cb(:,:),ab(:)
  integer::input,output,cases,c,n,p,t,vector,lasso,na,has_init,has_gram,has_xy,positive,i,backend
  character(1024)::file
  character(40)::prefix
  call get_command_argument(1,file);open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file);open(newunit=output,file=trim(file),status='replace')
  read(input,*)cases
  do c=1,cases
    read(input,*)n,p,t,vector,lasso,na,has_init,has_gram,has_xy
    read(input,*)options%l1_ratio,options%eps,options%n_alphas,options%tol,options%max_iter, &
      options%random_state,positive,options%selection,options%precompute
    options%positive=positive==1
    if(allocated(x))deallocate(x,y)
    if(allocated(grid))deallocate(grid)
    if(allocated(initial))deallocate(initial)
    if(allocated(gram))deallocate(gram)
    if(allocated(xy))deallocate(xy)
    allocate(x(n,p),y(n,t))
    do i=1,n
      read(input,*)x(i,:)
    end do
    do i=1,n
      read(input,*)y(i,:)
    end do
    if(na>=0)then
      allocate(grid(na));read(input,*)grid
      ab=grid
    end if
    if(has_init==1)then
      allocate(initial(t,p))
      do i=1,t
        read(input,*)initial(i,:)
      end do
      ib=initial
    end if
    if(has_gram==1)then
      allocate(gram(p,p))
      do i=1,p
        read(input,*)gram(i,:)
      end do
      gb=gram
    end if
    if(has_xy==1)then
      allocate(xy(p,t))
      do i=1,p
        read(input,*)xy(i,:)
      end do
      cb=xy
    end if
    xb=x;yb=y
    if(vector==1)then
      call vector_fit()
    else
      if(lasso==1)then
        fit=lasso_path(x,y,options,grid,initial,gram,xy)
      else
        fit=enet_path(x,y,options,grid,initial,gram,xy)
      end if
    end if
    call require(all(x==xb).and.all(y==yb),'coordinate path inputs changed')
    if(allocated(grid))call require(all(grid==ab),'coordinate path alpha input changed')
    if(allocated(initial))call require(all(initial==ib),'coordinate path initial coefficients changed')
    if(allocated(gram))call require(all(gram==gb),'coordinate path Gram input changed')
    if(allocated(xy))call require(all(xy==cb),'coordinate path correlations changed')
    write(prefix,'(a,i0,a)')'case',c,'_'
    call emit('shape',real(shape(fit%coefficients),dp))
    call emit('alphas',fit%alphas)
    call emit('coefficients',reshape(fit%coefficients,[size(fit%coefficients)]))
    call emit('gaps',fit%dual_gaps);call emit('iterations',real(fit%n_iter,dp))
    call emit('converged',real(merge(1,0,fit%converged),dp));call emit('tolerance',[fit%dual_tolerance])
    backend=0
    if(fit%backend=='gram')backend=1
    if(fit%backend=='multitask')backend=2
    call emit('backend',[real(backend,dp)]);call emit('preserved',[1._dp])
  end do
contains
  subroutine vector_fit()
    real(dp),allocatable::w(:),cross(:)
    if(allocated(initial))w=initial(1,:)
    if(allocated(xy))cross=xy(:,1)
    if(lasso==1)then
      fit=lasso_path(x,y(:,1),options,grid,w,gram,cross)
    else
      fit=enet_path(x,y(:,1),options,grid,w,gram,cross)
    end if
  end subroutine
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    write(output,'(a,*(1x,es25.16e3))')trim(prefix)//name,values
  end subroutine
end program
